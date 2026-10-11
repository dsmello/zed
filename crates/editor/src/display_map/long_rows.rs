use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    num::NonZeroU32,
    ops::Range,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use collections::{FxHasher, HashMap};
use gpui::{
    AppContext as _, Context, Font, FontId, HighlightStyle, LineLayout, Pixels, Task, TextAlign,
    WindowTextSystem,
};
use language::LanguageAwareStyling;
use multi_buffer::{Anchor, MultiBufferOffset, MultiBufferRow, MultiBufferSnapshot, ToOffset};
use parking_lot::Mutex;
use project::InlayId;
use sum_tree::{Bias, TreeMap};
use text::BufferId;

use crate::{
    EditorStyle, MAX_LINE_LEN,
    hover_links::InlayHighlight,
    scroll::{ScrollOffset, ScrollPixelOffset},
};

use super::{
    DisplayMap, DisplayPoint, DisplayRow, DisplaySnapshot, FoldPoint, HighlightKey,
    HighlightStyleInterner, Highlights,
    block_map::BlockSnapshot,
    fold_map::{FoldEdit, FoldSnapshot},
    inlay_map::has_chunk_renderer,
    row_ruler::{
        ComposedLayout, MAX_IDLE_REFRESHES, MAX_SYNC_RULER_LEN, MappedRow, RowEdit, RowRuler,
        RowRulerCache, RulerCacheVersion, RulerRequest, RulerShaper, TrailingWhitespace,
        renderer_metrics_key,
    },
    tab_map::{TabEdit, TabMap, TabPoint, TabSnapshot},
};

const MAX_TRACKED_RULER_DIRT: usize = 64;
const MAX_TRACKED_BUFFER_EDITS: usize = 16_384;
const MAX_PENDING_TAB_EDITS: usize = 256;
const MAX_CONCURRENT_RULER_BUILDS: usize = 2;
const MAX_ESTIMATED_WINDOW_EXTENSIONS: u32 = 4;
const ESTIMATED_WINDOW_EXTENSION_SLACK: ScrollPixelOffset = 1.25;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GridCell {
    pub(crate) width: Pixels,
    pub(crate) monospace: bool,
}

impl GridCell {
    const FIT_TOLERANCE: ScrollPixelOffset = 0.5;

    pub(crate) fn measure(
        text_system: &WindowTextSystem,
        style: &EditorStyle,
        snapshot: &DisplaySnapshot,
        font_size: Pixels,
    ) -> Self {
        let base = style.text.font();
        let font_id = text_system.resolve_font(&base);
        let width = text_system.em_layout_width(font_id, font_size);
        if snapshot
            .row_rulers
            .is_grid_rejected(renderer_metrics_key(&base, font_size))
        {
            return Self {
                width,
                monospace: false,
            };
        }
        let mut variants = vec![(base.weight, base.style)];
        let styles = style
            .syntax
            .highlights()
            .copied()
            .chain([
                style.inlay_hints_style,
                style.edit_prediction_styles.insertion,
                style.edit_prediction_styles.whitespace,
            ])
            .chain(snapshot.highlight_styles());
        for highlight in styles {
            let variant = (
                highlight.font_weight.unwrap_or(base.weight),
                highlight.font_style.unwrap_or(base.style),
            );
            if !variants.contains(&variant) {
                variants.push(variant);
            }
        }
        let monospace = width > Pixels::ZERO
            && base
                .features
                .tag_value_list()
                .iter()
                .all(|(_, value)| *value == 0)
            && variants.into_iter().all(|(weight, style)| {
                let font = Font {
                    weight,
                    style,
                    ..base.clone()
                };
                snapshot.row_rulers.has_uniform_ascii(
                    uniform_ascii_key(text_system.resolve_font(&font), font_size, width),
                    || has_uniform_ascii_advances(text_system, &font, font_size, width),
                )
            });
        Self { width, monospace }
    }

    pub(crate) fn fits(&self, columns: &Range<u32>, shaped_width: Pixels) -> bool {
        let grid_width = ScrollPixelOffset::from(self.width) * columns.len() as ScrollPixelOffset;
        (ScrollPixelOffset::from(shaped_width) - grid_width).abs() <= Self::FIT_TOLERANCE
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HorizontalViewport {
    pub(crate) scroll_columns: ScrollOffset,
    pub(crate) visible_columns: ScrollOffset,
    pub(crate) text_align: TextAlign,
    pub(crate) content_width: Pixels,
}

impl HorizontalViewport {
    pub(crate) fn aligned(&self, line_width: ScrollPixelOffset, cell: GridCell) -> Self {
        let content_width = ScrollPixelOffset::from(self.content_width);
        let alignment_offset = match self.text_align {
            TextAlign::Left => 0.,
            TextAlign::Center => (content_width - line_width) / 2.,
            TextAlign::Right => content_width - line_width,
        };
        if alignment_offset == 0. || cell.width <= Pixels::ZERO {
            return *self;
        }
        Self {
            scroll_columns: self.scroll_columns
                - alignment_offset / ScrollPixelOffset::from(cell.width),
            ..*self
        }
    }

    fn first_column(&self) -> usize {
        self.scroll_columns.max(0.).floor() as usize
    }

    fn column_count(&self) -> usize {
        (self.visible_columns.max(0.).ceil() as usize).max(1)
    }

    pub(crate) fn shaping_window(&self, row_len: u32) -> Range<u32> {
        let visible = self.column_count();
        let leading = visible / 2;
        let total = leading + visible * 2;
        let row_len = row_len as usize;
        let start = (self.first_column().saturating_sub(leading) / visible * visible)
            .min(row_len.saturating_sub(total));
        let end = (start + total).min(row_len);
        column_from_usize(start)..column_from_usize(end)
    }
}

#[derive(Clone)]
pub(crate) struct RuledRow {
    snapshot: Arc<DisplaySnapshot>,
    row: DisplayRow,
    pub(crate) ruler: Arc<RowRuler>,
    shaper: RulerShaper,
    pub(crate) exact: bool,
}

impl Debug for RuledRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RuledRow")
            .field("row", &self.row)
            .field("ruler", &self.ruler)
            .finish()
    }
}

impl RuledRow {
    pub(crate) fn columns_for_viewport(
        &self,
        viewport: &HorizontalViewport,
        cell: GridCell,
    ) -> Range<u32> {
        let viewport = viewport.aligned(self.ruler.width(), cell);
        let cell_width = ScrollPixelOffset::from(cell.width);
        let visible_width = viewport.visible_columns.max(1.) * cell_width;
        let left = viewport.scroll_columns.max(0.) * cell_width;
        self.ruler
            .columns_for_x_range(left - visible_width / 2.0..left + visible_width * 1.5)
    }

    fn x_for_column(&self, column: u32) -> ScrollPixelOffset {
        self.ruler
            .x_for_column(column, &self.snapshot, self.row, &self.shaper)
    }

    fn column_for_x(&self, x: ScrollPixelOffset) -> u32 {
        self.ruler
            .column_for_x(x, &self.snapshot, self.row, &self.shaper)
    }
}

#[derive(Clone)]
pub(crate) struct EstimatedRow {
    snapshot: Arc<DisplaySnapshot>,
    row: DisplayRow,
    row_len: u32,
    row_cells: u32,
}

impl Debug for EstimatedRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EstimatedRow")
            .field("row", &self.row)
            .field("row_len", &self.row_len)
            .field("row_cells", &self.row_cells)
            .finish()
    }
}

impl EstimatedRow {
    pub(crate) fn new(snapshot: &DisplaySnapshot, row: DisplayRow, row_len: u32) -> Self {
        Self {
            snapshot: snapshot.shared_snapshot(),
            row,
            row_len,
            row_cells: Self::cells_until(snapshot, row, row_len),
        }
    }

    fn cells_until(snapshot: &DisplaySnapshot, row: DisplayRow, column: u32) -> u32 {
        let start = snapshot.unwrapped_tab_point(DisplayPoint::new(row, 0), Bias::Left);
        let end = snapshot.unwrapped_tab_point(DisplayPoint::new(row, column), Bias::Left);
        snapshot
            .tab_snapshot()
            .text_summary_for_range(start..end)
            .first_line_chars
    }

    fn cells_before(&self, column: u32) -> u32 {
        if column >= self.row_len {
            return self.row_cells;
        }
        let column = self
            .snapshot
            .clip_window_column(self.row, column, self.row_len, Bias::Left);
        Self::cells_until(&self.snapshot, self.row, column)
    }

    fn column_at_cells(&self, cells: u32) -> u32 {
        if cells >= self.row_cells {
            return self.row_len;
        }
        let clip = |column| {
            self.snapshot
                .clip_window_column(self.row, column, self.row_len, Bias::Right)
        };
        let (mut low, mut high) = (0, self.row_len);
        while low < high {
            let middle = low + (high - low) / 2;
            if Self::cells_until(&self.snapshot, self.row, clip(middle)) >= cells {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        clip(low)
    }
}

#[derive(Clone, Debug)]
pub struct WindowedRowGeometry {
    row_len: u32,
    cell: GridCell,
    window: Range<u32>,
    ruled: Option<RuledRow>,
    estimated: Option<(EstimatedRow, Range<ScrollPixelOffset>)>,
}

impl WindowedRowGeometry {
    pub(crate) fn new(row_len: u32, cell: GridCell, window: Range<u32>) -> Self {
        Self {
            row_len,
            cell,
            window,
            ruled: None,
            estimated: None,
        }
    }

    pub(crate) fn ruled(ruled: RuledRow, row_len: u32, cell: GridCell, window: Range<u32>) -> Self {
        Self {
            ruled: Some(ruled),
            ..Self::new(row_len, cell, window)
        }
    }

    pub(crate) fn estimated(estimated: EstimatedRow, cell: GridCell, window: Range<u32>) -> Self {
        let cell_width = ScrollPixelOffset::from(cell.width);
        let window_x = cell_width * estimated.cells_before(window.start) as ScrollPixelOffset
            ..cell_width * estimated.cells_before(window.end) as ScrollPixelOffset;
        let row_len = estimated.row_len;
        Self {
            estimated: Some((estimated, window_x)),
            ..Self::new(row_len, cell, window)
        }
    }

    pub(crate) fn is_ruled(&self) -> bool {
        self.ruled.is_some()
    }

    pub(crate) fn is_provisional(&self) -> bool {
        self.estimated.is_some() || self.ruled.as_ref().is_some_and(|ruled| !ruled.exact)
    }

    pub(crate) fn row_len(&self) -> u32 {
        self.row_len
    }

    pub(crate) fn window(&self) -> &Range<u32> {
        &self.window
    }

    pub(crate) fn cell(&self) -> GridCell {
        self.cell
    }

    pub(crate) fn start_x(&self) -> ScrollPixelOffset {
        match (&self.ruled, &self.estimated) {
            (Some(ruled), _) => ruled.ruler.x_range_for_columns(self.window.clone()).start,
            (None, Some((_, window_x))) => window_x.start,
            (None, None) => self.column_x(self.window.start),
        }
    }

    pub(crate) fn end_x(&self) -> ScrollPixelOffset {
        match (&self.ruled, &self.estimated) {
            (Some(ruled), _) => ruled.ruler.x_range_for_columns(self.window.clone()).end,
            (None, Some((_, window_x))) => window_x.end,
            (None, None) => self.column_x(self.window.end),
        }
    }

    pub(crate) fn width(&self, shaped_width: Pixels) -> ScrollPixelOffset {
        let row_width = self.column_x(self.row_len);
        if self.ruled.is_some() {
            return row_width;
        }
        row_width.max(self.start_x() + ScrollPixelOffset::from(shaped_width))
    }

    pub(crate) fn column_x(&self, column: u32) -> ScrollPixelOffset {
        if let Some(ruled) = &self.ruled {
            return ruled.x_for_column(column.min(self.row_len));
        }
        let cells = match &self.estimated {
            Some((estimated, _)) => estimated.cells_before(column),
            None => column,
        };
        ScrollPixelOffset::from(self.cell.width) * cells as ScrollPixelOffset
    }

    pub(crate) fn column_for_x(&self, x: ScrollPixelOffset) -> u32 {
        if let Some(ruled) = &self.ruled {
            return ruled.column_for_x(x).min(self.row_len);
        }
        if self.cell.width <= Pixels::ZERO {
            return 0;
        }
        let cells = (x / ScrollPixelOffset::from(self.cell.width))
            .round()
            .max(0.);
        let cells = column_from_usize(cells as usize);
        match &self.estimated {
            Some((estimated, _)) => estimated.column_at_cells(cells),
            None => cells.min(self.row_len),
        }
    }

    pub(crate) fn reaches_left_edge(&self) -> bool {
        self.window.start == 0
    }

    pub(crate) fn reaches_right_edge(&self) -> bool {
        self.window.end >= self.row_len
    }

    pub(crate) fn column_left_of_window_for_x(&self, x: ScrollPixelOffset) -> u32 {
        self.column_for_x(x).min(self.window.start)
    }

    pub(crate) fn column_right_of_window_for_x(&self, x: ScrollPixelOffset) -> u32 {
        self.column_for_x(x).max(self.window.end)
    }
}

pub enum RowLayout {
    Shaped(Arc<LineLayout>),
    Windowed {
        geometry: WindowedRowGeometry,
        shaped: WindowText,
    },
}

pub struct WindowText(ComposedLayout);

impl WindowText {
    pub(crate) fn empty() -> Self {
        Self(ComposedLayout::default())
    }

    pub(crate) fn line(layout: Arc<LineLayout>) -> Self {
        Self(ComposedLayout::from_line(layout))
    }

    pub(crate) fn composed(layout: ComposedLayout) -> Self {
        Self(layout)
    }

    fn width(&self) -> Pixels {
        self.0.width()
    }

    fn x_for_index(&self, index: usize) -> Pixels {
        self.0.x_for_index(index)
    }

    fn closest_index_for_x(&self, x: Pixels) -> usize {
        self.0.closest_index_for_x(x)
    }
}

impl RowLayout {
    pub fn width(&self) -> ScrollPixelOffset {
        match self {
            Self::Shaped(layout) => ScrollPixelOffset::from(layout.width),
            Self::Windowed { geometry, shaped } => geometry.width(shaped.width()),
        }
    }

    pub(crate) fn x_for_index(&self, index: usize) -> ScrollPixelOffset {
        match self {
            Self::Shaped(layout) => ScrollPixelOffset::from(layout.x_for_index(index)),
            Self::Windowed { geometry, shaped } => {
                let window = geometry.window();
                let column = column_from_usize(index);
                if !window.is_empty() && column >= window.start && column <= window.end {
                    geometry.start_x()
                        + ScrollPixelOffset::from(
                            shaped.x_for_index((column - window.start) as usize),
                        )
                } else {
                    geometry.column_x(column)
                }
            }
        }
    }

    pub fn closest_index_for_x(&self, x: ScrollPixelOffset) -> usize {
        match self {
            Self::Shaped(layout) => layout.closest_index_for_x(Pixels::from(x)),
            Self::Windowed { geometry, shaped } => {
                let start_x = geometry.start_x();
                let window = geometry.window();
                if x < start_x {
                    geometry.column_left_of_window_for_x(x) as usize
                } else if x <= start_x + ScrollPixelOffset::from(shaped.width()) {
                    window.start as usize + shaped.closest_index_for_x(Pixels::from(x - start_x))
                } else {
                    geometry.column_right_of_window_for_x(x) as usize
                }
            }
        }
    }
}

impl DisplaySnapshot {
    #[cfg(test)]
    pub(crate) fn is_windowed_row(&self, display_row: DisplayRow, cell: GridCell) -> bool {
        self.is_long_unwrapped_row(display_row) && self.row_has_exact_grid(display_row, cell)
    }

    pub(crate) fn is_long_unwrapped_row(&self, display_row: DisplayRow) -> bool {
        self.long_unwrapped_row_len(display_row).is_some()
    }

    pub(crate) fn long_unwrapped_row_len(&self, display_row: DisplayRow) -> Option<u32> {
        if self.masked
            || self.has_soft_wraps()
            || display_row > self.max_point().row()
            || self.is_block_line(display_row)
        {
            return None;
        }
        let row_len = self.line_len(display_row);
        (row_len as usize > MAX_LINE_LEN).then_some(row_len)
    }

    pub(crate) fn grid_window(
        &self,
        display_row: DisplayRow,
        row_len: u32,
        viewport: &HorizontalViewport,
        cell: GridCell,
        shaper: &RulerShaper,
    ) -> Option<(Range<u32>, Arc<LineLayout>)> {
        if !self.row_has_exact_grid(display_row, cell) {
            return None;
        }
        let row_width = ScrollPixelOffset::from(cell.width) * row_len as ScrollPixelOffset;
        let window = viewport.aligned(row_width, cell).shaping_window(row_len);
        let shaped = shaper.layout_columns(self, display_row, window.clone());
        if !cell.fits(&window, shaped.width) {
            self.row_rulers.reject_grid(renderer_metrics_key(
                &shaper.style.text.font(),
                shaper.font_size,
            ));
            return None;
        }
        Some((window, shaped))
    }

    pub(crate) fn ruled_row(
        &self,
        display_row: DisplayRow,
        shaper: RulerShaper,
    ) -> Option<RuledRow> {
        let tab_row = self
            .unwrapped_tab_point(DisplayPoint::new(display_row, 0), Bias::Left)
            .row();
        let (ruler, exact) = self
            .current_ruler(tab_row, display_row, &shaper)
            .or_else(|| self.build_ruler(tab_row, display_row, &shaper))?;
        Some(RuledRow {
            snapshot: self.shared_snapshot(),
            row: display_row,
            ruler,
            shaper,
            exact,
        })
    }

    pub(crate) fn covering_estimated_window<T>(
        &self,
        estimated: &EstimatedRow,
        viewport: &HorizontalViewport,
        cell: GridCell,
        mut shape: impl FnMut(Range<u32>) -> (T, Pixels),
    ) -> (Range<u32>, T) {
        let mut columns = self.estimated_window(estimated, viewport, cell);
        let bound_end = self.window_bound_end(estimated, &columns);
        let (mut shaped, mut width) = shape(columns.clone());
        let cell_width = ScrollPixelOffset::from(cell.width);
        let row_width = cell_width * estimated.row_cells as ScrollPixelOffset;
        let viewport = viewport.aligned(row_width, cell);
        let visible_end =
            (viewport.scroll_columns.max(0.) + viewport.visible_columns.max(1.)) * cell_width;
        let start_x = cell_width * estimated.cells_before(columns.start) as ScrollPixelOffset;
        for _ in 0..MAX_ESTIMATED_WINDOW_EXTENSIONS {
            let shaped_width = ScrollPixelOffset::from(width);
            if columns.end >= bound_end || start_x + shaped_width >= visible_end {
                break;
            }
            let factor = if shaped_width > 0. {
                ((visible_end - start_x) / shaped_width * ESTIMATED_WINDOW_EXTENSION_SLACK)
                    .ceil()
                    .min(u32::MAX as ScrollPixelOffset) as u32
            } else {
                u32::MAX
            };
            columns.end = self
                .extended_window_end(estimated, &columns, factor.max(2))
                .min(bound_end);
            (shaped, width) = shape(columns.clone());
        }
        (columns, shaped)
    }

    pub(crate) fn estimated_window_bound(
        &self,
        estimated: &EstimatedRow,
        viewport: &HorizontalViewport,
        cell: GridCell,
    ) -> Range<u32> {
        let columns = self.estimated_window(estimated, viewport, cell);
        let end = self.window_bound_end(estimated, &columns);
        columns.start..end
    }

    fn window_bound_end(&self, estimated: &EstimatedRow, columns: &Range<u32>) -> u32 {
        let mut columns = columns.clone();
        for _ in 0..MAX_ESTIMATED_WINDOW_EXTENSIONS {
            columns.end = self.extended_window_end(estimated, &columns, 2);
        }
        columns.end
    }

    fn extended_window_end(
        &self,
        estimated: &EstimatedRow,
        columns: &Range<u32>,
        factor: u32,
    ) -> u32 {
        let len = (columns.end - columns.start).max(1);
        let end = columns
            .start
            .saturating_add(len.saturating_mul(factor))
            .min(estimated.row_len);
        self.clip_window_column(estimated.row, end, estimated.row_len, Bias::Right)
            .clamp(columns.end, estimated.row_len)
    }

    fn clip_window_column(&self, row: DisplayRow, column: u32, row_len: u32, bias: Bias) -> u32 {
        let point = DisplayPoint::new(row, column);
        let clipped = self
            .inlay_snapshot()
            .clip_point_inside_text_inlay(self.display_point_to_inlay_point(point, bias), bias)
            .map(|inlay_point| {
                self.fold_point_to_display_point(
                    self.fold_snapshot().to_fold_point(inlay_point, bias),
                )
            })
            .unwrap_or_else(|| self.clip_ignoring_line_ends(point, bias));
        if clipped.row() == row {
            clipped.column().min(row_len)
        } else if clipped.row() > row {
            row_len
        } else {
            0
        }
    }

    pub(crate) fn estimated_window(
        &self,
        estimated: &EstimatedRow,
        viewport: &HorizontalViewport,
        cell: GridCell,
    ) -> Range<u32> {
        let row_width =
            ScrollPixelOffset::from(cell.width) * estimated.row_cells as ScrollPixelOffset;
        let cells = viewport
            .aligned(row_width, cell)
            .shaping_window(estimated.row_cells);
        let start = estimated.column_at_cells(cells.start);
        let end = estimated.column_at_cells(cells.end);
        start..end.max(start)
    }

    pub(crate) fn cached_trailing_whitespace_start(
        &self,
        display_row: DisplayRow,
        compute: impl FnOnce() -> usize,
    ) -> usize {
        let tab_row = self
            .unwrapped_tab_point(DisplayPoint::new(display_row, 0), Bias::Left)
            .row();
        self.row_rulers
            .trailing_whitespace_start(tab_row, self.line_len(display_row), compute)
    }

    pub(crate) fn row_has_exact_grid(&self, display_row: DisplayRow, cell: GridCell) -> bool {
        if !cell.monospace {
            return false;
        }
        let fold_snapshot = self.fold_snapshot();
        let fold_row = self
            .display_point_to_fold_point(DisplayPoint::new(display_row, 0), Bias::Left)
            .row();
        let fold_range =
            FoldPoint::new(fold_row, 0)..FoldPoint::new(fold_row, fold_snapshot.line_len(fold_row));
        let summary = fold_snapshot.text_summary_for_range(fold_range.clone());
        if summary.chars != summary.len.0 {
            return false;
        }
        let inlay_range = fold_range.start.to_inlay_point(fold_snapshot)
            ..fold_range.end.to_inlay_point(fold_snapshot);
        let inlay_snapshot = self.inlay_snapshot();
        let buffer_range = inlay_snapshot.to_buffer_point(inlay_range.start)
            ..inlay_snapshot.to_buffer_point(inlay_range.end);
        let buffer = self.buffer_snapshot();
        let buffer_offsets =
            buffer.point_to_offset(buffer_range.start)..buffer.point_to_offset(buffer_range.end);
        fold_snapshot.folds_in_range(buffer_range).next().is_none()
            && !inlay_snapshot.has_inlays_matching(inlay_range, |inlay, text_range| {
                has_chunk_renderer(inlay)
                    || !inlay
                        .text()
                        .chunks_in_range(text_range)
                        .all(|chunk| all_grid_bytes(chunk.as_bytes()))
            })
            && !self.control_rows.contains_control(buffer, buffer_offsets)
    }

    fn current_ruler(
        &self,
        tab_row: u32,
        display_row: DisplayRow,
        shaper: &RulerShaper,
    ) -> Option<(Arc<RowRuler>, bool)> {
        let rulers = &self.row_rulers;
        let (ruler, verified) = rulers.current(tab_row, shaper)?;
        if !verified && !self.ruler_stamps_match(tab_row, &ruler) {
            rulers.retire(tab_row);
            return None;
        }
        if !ruler.needs_measuring() {
            if !verified {
                rulers.set_current(tab_row, ruler.clone());
            }
            return Some((ruler, true));
        }
        if ruler.len() > MAX_SYNC_RULER_LEN && ruler.unmeasured_len() > MAX_SYNC_RULER_LEN {
            rulers.request(tab_row, || {
                RulerRequest::measure(
                    self.shared_snapshot(),
                    display_row,
                    shaper.clone(),
                    ruler.clone(),
                )
            });
            rulers.retire(tab_row);
            return None;
        }
        match ruler.measured(self, display_row, shaper, &rulers.renderer_widths(shaper)) {
            Some(measured) => {
                let measured = Arc::new(measured);
                rulers.set_current(tab_row, measured.clone());
                Some((measured, true))
            }
            None => {
                rulers.retire(tab_row);
                None
            }
        }
    }

    fn build_ruler(
        &self,
        tab_row: u32,
        display_row: DisplayRow,
        shaper: &RulerShaper,
    ) -> Option<(Arc<RowRuler>, bool)> {
        let rulers = &self.row_rulers;
        let previous = rulers.previous(tab_row, shaper);
        let row_len = self.line_len(display_row);
        if row_len <= MAX_SYNC_RULER_LEN {
            let ruler = Arc::new(RowRuler::new(
                self,
                display_row,
                shaper,
                previous.as_ref().map(|previous| previous.ruler.as_ref()),
                &rulers.renderer_widths(shaper),
            ));
            rulers.set_current(tab_row, ruler.clone());
            return Some((ruler, true));
        }
        let interim = previous
            .as_ref()
            .filter(|previous| previous.matches_text)
            .map(|previous| &previous.ruler)
            .filter(|previous| previous.len() == row_len)
            .and_then(|previous| {
                if !previous.needs_measuring() {
                    return Some(previous.clone());
                }
                if previous.unmeasured_len() > MAX_SYNC_RULER_LEN {
                    return None;
                }
                let measured = Arc::new(previous.measured(
                    self,
                    display_row,
                    shaper,
                    &rulers.renderer_widths(shaper),
                )?);
                rulers.set_previous(tab_row, measured.clone());
                Some(measured)
            });
        rulers.request(tab_row, || {
            RulerRequest::new(
                self.shared_snapshot(),
                display_row,
                shaper.clone(),
                previous.map(|previous| previous.ruler),
            )
        });
        interim.map(|interim| (interim, false))
    }

    fn ruler_stamps_match(&self, tab_row: u32, ruler: &RowRuler) -> bool {
        let buffer = self.buffer_snapshot();
        let range = row_buffer_range(self.fold_snapshot(), tab_row);
        ruler.diagnostics() == diagnostic_keys(buffer, range.clone()).as_slice()
            && ruler
                .syntax()
                .is_none_or(|stamp| stamp.still_applies(&SyntaxStamp::new(buffer, range)))
    }

    fn shared_snapshot(&self) -> Arc<DisplaySnapshot> {
        self.ruled_row_snapshot
            .get_or_init(|| Arc::new(self.clone()))
            .clone()
    }
}

pub(super) struct RecordedTabMap {
    tab_map: TabMap,
    pending: Option<Vec<Vec<TabEdit>>>,
    pending_len: usize,
}

impl RecordedTabMap {
    pub(super) fn new(tab_map: TabMap) -> Self {
        Self {
            tab_map,
            pending: Some(Vec::new()),
            pending_len: 0,
        }
    }

    pub(super) fn sync(
        &mut self,
        fold_snapshot: FoldSnapshot,
        fold_edits: Vec<FoldEdit>,
        tab_size: NonZeroU32,
    ) -> (TabSnapshot, Vec<TabEdit>) {
        let (snapshot, edits) = self.tab_map.sync(fold_snapshot, fold_edits, tab_size);
        if !edits.is_empty()
            && let Some(pending) = &mut self.pending
        {
            self.pending_len += edits.len();
            if self.pending_len > MAX_PENDING_TAB_EDITS {
                self.pending = None;
            } else {
                pending.push(edits.clone());
            }
        }
        (snapshot, edits)
    }

    fn take_edits(&mut self) -> Option<Vec<Vec<TabEdit>>> {
        self.pending_len = 0;
        std::mem::replace(&mut self.pending, Some(Vec::new()))
    }
}

pub(super) struct RulerBuild {
    id: usize,
    tab_row: u32,
    row_len: u32,
    metrics_fingerprint: u64,
    edits: Vec<Vec<RowEdit>>,
    last_requested: u64,
    cancelled: Arc<AtomicBool>,
    _task: Task<()>,
}

impl RulerBuild {
    pub(super) fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }
}

impl Drop for RulerBuild {
    fn drop(&mut self) {
        self.cancel();
    }
}

#[derive(Default)]
pub(super) struct RulerDirt {
    pub(super) styles: DirtyRanges,
    pub(super) buffer_edits: DirtyRanges,
}

#[derive(Default)]
pub(super) struct DirtyRanges {
    all: bool,
    ranges: Vec<Range<Anchor>>,
}

impl DirtyRanges {
    pub(super) fn mark(&mut self, ranges: impl IntoIterator<Item = Range<Anchor>>) {
        self.mark_up_to(ranges, MAX_TRACKED_RULER_DIRT);
    }

    pub(super) fn mark_edits(&mut self, ranges: impl IntoIterator<Item = Range<Anchor>>) {
        self.mark_up_to(ranges, MAX_TRACKED_BUFFER_EDITS);
    }

    fn mark_up_to(&mut self, ranges: impl IntoIterator<Item = Range<Anchor>>, limit: usize) {
        if self.all {
            return;
        }
        self.ranges.extend(ranges);
        if self.ranges.len() > limit {
            self.mark_all();
        }
    }

    pub(super) fn mark_all(&mut self) {
        self.all = true;
        self.ranges.clear();
    }
}

#[derive(Clone, Debug)]
pub(super) struct SyntaxStamp {
    versions: Vec<(BufferId, usize)>,
    unstyled: bool,
}

impl SyntaxStamp {
    pub(super) fn new(buffer: &MultiBufferSnapshot, range: Range<MultiBufferOffset>) -> Self {
        let ranges = buffer.range_to_buffer_ranges(range);
        Self {
            versions: ranges
                .iter()
                .map(|(buffer, ..)| (buffer.remote_id(), buffer.syntax_update_count()))
                .collect(),
            unstyled: ranges.iter().all(|(buffer, range, _)| {
                buffer
                    .syntax_layers_for_range(range.start.0..range.end.0, true)
                    .next()
                    .is_none()
            }),
        }
    }

    fn still_applies(&self, current: &Self) -> bool {
        let same_buffers = || {
            self.versions
                .iter()
                .map(|(buffer_id, _)| buffer_id)
                .eq(current.versions.iter().map(|(buffer_id, _)| buffer_id))
        };
        self.versions == current.versions || self.unstyled && current.unstyled && same_buffers()
    }
}

pub(super) type DiagnosticKey = (usize, usize, lsp::DiagnosticSeverity, bool, bool);

pub(super) fn diagnostic_keys(
    buffer: &MultiBufferSnapshot,
    range: Range<MultiBufferOffset>,
) -> Vec<DiagnosticKey> {
    buffer
        .diagnostics_in_range(range.clone())
        .map(|entry| {
            (
                entry.range.start.0.wrapping_sub(range.start.0),
                entry.range.end.0.wrapping_sub(range.start.0),
                entry.diagnostic.severity,
                entry.diagnostic.underline,
                entry.diagnostic.is_unnecessary,
            )
        })
        .collect()
}

#[derive(Default)]
pub(super) struct ControlRows(Mutex<HashMap<MultiBufferOffset, ControlRow>>);

struct ControlRow {
    range: Range<Anchor>,
    end: MultiBufferOffset,
    control: Option<Anchor>,
    idle_refreshes: u8,
}

impl ControlRows {
    fn edited(
        &self,
        buffer: &MultiBufferSnapshot,
        mut dirty: Vec<Range<MultiBufferOffset>>,
    ) -> Self {
        for range in &mut dirty {
            if range.end < range.start {
                *range = range.end..range.start;
            }
        }
        dirty.sort_unstable_by_key(|range| range.start);
        let dirty = dirty.into_iter().fold(
            Vec::<Range<MultiBufferOffset>>::new(),
            |mut merged, range| {
                match merged.last_mut() {
                    Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                    _ => merged.push(range),
                }
                merged
            },
        );
        let rows = self
            .0
            .lock()
            .values()
            .filter(|row| row.idle_refreshes < MAX_IDLE_REFRESHES)
            .filter_map(|row| {
                let start = row.range.start.to_offset(buffer);
                let start_point = buffer.offset_to_point(start);
                if start_point.column != 0 {
                    return None;
                }
                let scanned_end = row.range.end.to_offset(buffer);
                let end = start + buffer.line_len(MultiBufferRow(start_point.row)) as usize;
                let first_touched = dirty.partition_point(|dirty| dirty.end < start);
                let mut unscanned = dirty[first_touched..]
                    .iter()
                    .take_while(|dirty| dirty.start <= end)
                    .map(|dirty| dirty.start.max(start)..dirty.end.min(end))
                    .collect::<Vec<_>>();
                let mut known_control = None;
                if let Some(offset) = row
                    .control
                    .map(|control| control.to_offset(buffer))
                    .filter(|offset| *offset < end)
                {
                    if offset >= start && is_control_at(buffer, offset) {
                        known_control = Some(offset);
                    } else {
                        unscanned.push(offset.max(start)..end);
                    }
                }
                if scanned_end < end {
                    unscanned.push(scanned_end.max(start)..end);
                }
                unscanned.sort_unstable_by_key(|range| range.start);
                let control = unscanned
                    .into_iter()
                    .take_while(|range| known_control.is_none_or(|known| range.start < known))
                    .find_map(|range| {
                        first_control(
                            buffer,
                            range.start
                                ..known_control.map_or(range.end, |known| range.end.min(known)),
                        )
                    })
                    .or(known_control);
                Some((
                    start,
                    ControlRow {
                        range: buffer.anchor_before(start)..buffer.anchor_after(end),
                        end,
                        control: control.map(|offset| buffer.anchor_after(offset)),
                        idle_refreshes: row.idle_refreshes + 1,
                    },
                ))
            })
            .collect::<HashMap<_, _>>();
        Self(Mutex::new(rows))
    }

    fn contains_control(
        &self,
        buffer: &MultiBufferSnapshot,
        range: Range<MultiBufferOffset>,
    ) -> bool {
        if range.is_empty() {
            return false;
        }
        if let Some(row) = self.0.lock().get_mut(&range.start)
            && row.end == range.end
        {
            row.idle_refreshes = 0;
            return row.control.is_some();
        }
        let control = first_control(buffer, range.clone());
        self.0.lock().insert(
            range.start,
            ControlRow {
                range: buffer.anchor_before(range.start)..buffer.anchor_after(range.end),
                end: range.end,
                control: control.map(|offset| buffer.anchor_after(offset)),
                idle_refreshes: 0,
            },
        );
        control.is_some()
    }
}

fn first_control(
    buffer: &MultiBufferSnapshot,
    range: Range<MultiBufferOffset>,
) -> Option<MultiBufferOffset> {
    let mut offset = range.start;
    for bytes in buffer.bytes_in_range(range) {
        if !all_grid_bytes(bytes) {
            return bytes
                .iter()
                .position(|byte| !is_grid_byte(*byte))
                .map(|ix| offset + ix);
        }
        offset += bytes.len();
    }
    None
}

fn is_control_at(buffer: &MultiBufferSnapshot, offset: MultiBufferOffset) -> bool {
    buffer
        .bytes_in_range(offset..offset + 1usize)
        .next()
        .and_then(|bytes| bytes.first())
        .is_some_and(|byte| !is_grid_byte(*byte))
}

pub(super) fn affects_shaping(style: &HighlightStyle) -> bool {
    style.color.is_some()
        || style.font_weight.is_some()
        || style.font_style.is_some()
        || style.underline.is_some()
        || style.strikethrough.is_some()
        || style.fade_out.is_some()
}

impl DisplayMap {
    pub(crate) fn spawn_ruler_builds(
        &mut self,
        focus_row: Option<DisplayRow>,
        cx: &mut Context<Self>,
    ) {
        self.ruler_spawn_generation += 1;
        let generation = self.ruler_spawn_generation;
        let mut requests = self
            .ruler_requests
            .take()
            .into_iter()
            .filter_map(|(tab_row, request)| {
                let metrics_fingerprint = request.shaper().metrics_fingerprint();
                match self.ruler_builds.iter_mut().find(|build| {
                    !build.is_cancelled()
                        && build.tab_row == tab_row
                        && build.metrics_fingerprint == metrics_fingerprint
                }) {
                    Some(build) => {
                        build.last_requested = generation;
                        None
                    }
                    None => Some((tab_row, metrics_fingerprint, request)),
                }
            })
            .collect::<Vec<_>>();
        for build in &self.ruler_builds {
            if build.last_requested != generation {
                build.cancel();
            }
        }
        requests.sort_unstable_by_key(|(tab_row, _, request)| {
            let distance = focus_row.map_or(0, |focus_row| request.row().0.abs_diff(focus_row.0));
            std::cmp::Reverse((distance, *tab_row))
        });
        while self.ruler_builds.len() < MAX_CONCURRENT_RULER_BUILDS
            && let Some((tab_row, metrics_fingerprint, request)) = requests.pop()
        {
            let id = self.next_ruler_build_id;
            self.next_ruler_build_id += 1;
            let row_len = request.row_len();
            let renderer_widths = self.row_rulers.renderer_widths(request.shaper());
            let cancelled = Arc::new(AtomicBool::new(false));
            let build = cx.background_spawn({
                let cancelled = cancelled.clone();
                async move { request.build(&renderer_widths, &cancelled) }
            });
            let task = cx.spawn(async move |display_map, cx| {
                let ruler = build.await;
                display_map
                    .update(cx, |display_map, cx| {
                        display_map.finish_ruler_build(id, ruler, cx)
                    })
                    .ok();
            });
            self.ruler_builds.push(RulerBuild {
                id,
                tab_row,
                row_len,
                metrics_fingerprint,
                edits: Vec::new(),
                last_requested: generation,
                cancelled,
                _task: task,
            });
        }
        self.ruler_requests_deferred = !requests.is_empty();
    }

    pub(super) fn refresh_row_caches(&mut self, block_snapshot: &BlockSnapshot) {
        let tab_snapshot = &block_snapshot.wrap_snapshot.tab_snapshot;
        let buffer = &tab_snapshot.fold_snapshot.inlay_snapshot.buffer;
        let version = RulerCacheVersion {
            tabs: tab_snapshot.version,
            highlights: self.highlight_version,
            non_text_state: buffer.non_text_state_update_count(),
            diagnostics_max_severity: self.diagnostics_max_severity,
            tab_size: tab_snapshot.tab_size,
            masked: self.masked,
        };
        let tab_edits = self.tab_map.take_edits();
        let previous_version = self.row_rulers.version();
        if previous_version == version && tab_edits.as_ref().is_some_and(Vec::is_empty) {
            return;
        }
        let dirt = std::mem::take(&mut self.ruler_dirt);
        self.control_rows = if dirt.buffer_edits.all {
            Arc::default()
        } else {
            let edited = dirt
                .buffer_edits
                .ranges
                .iter()
                .map(|range| range.start.to_offset(buffer)..range.end.to_offset(buffer))
                .collect::<Vec<_>>();
            Arc::new(self.control_rows.edited(buffer, edited))
        };
        let tab_edits_tracked = tab_edits.is_some();
        let all_changed = dirt.styles.all
            || tab_edits.is_none()
            || previous_version.diagnostics_max_severity != version.diagnostics_max_severity
            || previous_version.tab_size != version.tab_size
            || previous_version.masked != version.masked;
        let tab_edits = tab_edits.unwrap_or_default();
        let diagnostics_changed = previous_version.non_text_state != version.non_text_state;
        let fold_snapshot = &tab_snapshot.fold_snapshot;
        let restyled_rows = dirt
            .styles
            .ranges
            .iter()
            .map(|range| {
                tab_row_for_offset(fold_snapshot, range.start.to_offset(buffer))
                    ..=tab_row_for_offset(fold_snapshot, range.end.to_offset(buffer))
            })
            .collect::<Vec<_>>();
        let is_restyled =
            |row: u32| all_changed || restyled_rows.iter().any(|rows| rows.contains(&row));
        let remap = |row: u32, ruler: &RowRuler, recently_used: bool| {
            if !tab_edits_tracked {
                return (MappedRow::Changed(row), false);
            }
            let mapped = map_row_through_tab_edits(row, ruler.len(), &tab_edits);
            let row = match &mapped {
                MappedRow::Changed(_) => return (mapped, false),
                MappedRow::Unchanged(row) | MappedRow::Edited { row, .. } => *row,
            };
            if !recently_used || is_restyled(row) {
                return (mapped, false);
            }
            if diagnostics_changed || ruler.syntax().is_some() {
                let range = row_buffer_range(fold_snapshot, row);
                if diagnostics_changed
                    && ruler.diagnostics() != diagnostic_keys(buffer, range.clone())
                {
                    return (mapped, false);
                }
                if ruler
                    .syntax()
                    .is_some_and(|stamp| !stamp.still_applies(&SyntaxStamp::new(buffer, range)))
                {
                    return (mapped, false);
                }
            }
            (mapped, true)
        };
        let remap_trailing_whitespace = |row: u32, trailing: TrailingWhitespace| {
            if !tab_edits_tracked {
                return None;
            }
            match map_row_through_tab_edits(row, trailing.row_len, &tab_edits) {
                MappedRow::Unchanged(row) => Some((row, trailing)),
                MappedRow::Edited { row, len, edits } => {
                    let (start, inserted) =
                        shift_trailing_whitespace_start(trailing.start, &edits)?;
                    let start = inserted
                        .and_then(|inserted| {
                            last_non_whitespace_end(
                                tab_snapshot,
                                row,
                                inserted.start.min(len)..inserted.end.min(len),
                            )
                        })
                        .map_or(start, |end| end.max(start));
                    Some((
                        row,
                        TrailingWhitespace {
                            row_len: len,
                            start,
                            ..trailing
                        },
                    ))
                }
                MappedRow::Changed(_) => None,
            }
        };
        self.ruler_requests = Arc::default();
        self.row_rulers = Arc::new(RowRulerCache::new(
            version,
            Some(&self.row_rulers),
            &self.ruler_requests,
            remap,
            remap_trailing_whitespace,
        ));
        for build in &mut self.ruler_builds {
            if build.is_cancelled() {
                continue;
            }
            let row = match map_row_through_tab_edits(build.tab_row, build.row_len, &tab_edits) {
                MappedRow::Unchanged(row) => row,
                MappedRow::Edited { row, len, edits } => {
                    build.row_len = len;
                    build.edits.extend(edits);
                    row
                }
                MappedRow::Changed(_) => {
                    build.cancel();
                    continue;
                }
            };
            build.tab_row = row;
            if is_restyled(row) {
                build.cancel();
            }
        }
    }

    pub(super) fn mark_all_rulers_dirty(&mut self) {
        self.ruler_dirt.styles.mark_all();
    }

    pub(super) fn mark_text_highlight_dirty(
        &mut self,
        key: HighlightKey,
        ranges: &[Range<Anchor>],
        style: HighlightStyle,
        merge: bool,
        buffer: &MultiBufferSnapshot,
    ) {
        let previous = self.text_highlights.get(&key);
        if !affects_shaping(&style)
            && !previous.is_some_and(|previous| affects_shaping(&previous.0))
        {
            return;
        }
        let previous_hull = previous
            .filter(|previous| !merge || previous.0 != style)
            .and_then(|previous| anchor_hull(&previous.1, buffer));
        let dirty = previous_hull
            .into_iter()
            .chain(anchor_hull(ranges, buffer))
            .collect::<Vec<_>>();
        self.highlight_version += 1;
        self.ruler_dirt.styles.mark(dirty);
    }

    pub(super) fn mark_inlay_highlight_dirty(
        &mut self,
        key: HighlightKey,
        highlights: &[InlayHighlight],
        style: HighlightStyle,
    ) {
        let previous_affects_shaping = self.inlay_highlights.get(&key).is_some_and(|highlights| {
            highlights
                .iter()
                .any(|(_, (style, _))| affects_shaping(style))
        });
        if !affects_shaping(&style) && !previous_affects_shaping {
            return;
        }
        self.highlight_version += 1;
        if self.has_multiline_inlays() {
            self.mark_all_rulers_dirty();
            return;
        }
        let dirty = highlights
            .iter()
            .map(|highlight| highlight.inlay_position..highlight.inlay_position)
            .collect::<Vec<_>>();
        self.ruler_dirt.styles.mark(dirty);
    }

    pub(super) fn mark_removed_highlights_dirty(
        &mut self,
        text: Option<&(HighlightStyle, Vec<Range<Anchor>>)>,
        inlays: Option<&TreeMap<InlayId, (HighlightStyle, InlayHighlight)>>,
    ) {
        if self.has_multiline_inlays()
            && inlays
                .is_some_and(|inlays| inlays.iter().any(|(_, (style, _))| affects_shaping(style)))
        {
            self.highlight_version += 1;
            self.mark_all_rulers_dirty();
            return;
        }
        let dirty = text
            .filter(|(style, _)| affects_shaping(style))
            .into_iter()
            .flat_map(|(_, ranges)| ranges.iter().cloned())
            .chain(inlays.into_iter().flat_map(|inlays| {
                inlays
                    .iter()
                    .filter(|(_, (style, _))| affects_shaping(style))
                    .map(|(_, (_, highlight))| highlight.inlay_position..highlight.inlay_position)
            }))
            .collect::<Vec<_>>();
        if !dirty.is_empty() {
            self.highlight_version += 1;
            self.ruler_dirt.styles.mark(dirty);
        }
    }

    pub(super) fn mark_semantic_styles_dirty(&mut self, interner: &HighlightStyleInterner) {
        if interner.styles().any(affects_shaping) {
            self.highlight_version += 1;
            self.mark_all_rulers_dirty();
        }
    }

    fn has_multiline_inlays(&self) -> bool {
        self.inlay_map
            .current_inlays()
            .any(|inlay| inlay.text().max_point().row > 0)
    }

    fn finish_ruler_build(&mut self, id: usize, ruler: Option<RowRuler>, cx: &mut Context<Self>) {
        let Some(ix) = self.ruler_builds.iter().position(|build| build.id == id) else {
            return;
        };
        let build = self.ruler_builds.swap_remove(ix);
        let ruler = ruler.filter(|_| !build.is_cancelled()).and_then(|ruler| {
            if build.edits.is_empty() {
                Some(ruler)
            } else {
                ruler.interpolate(&build.edits)
            }
        });
        let inserted =
            ruler.is_some_and(|ruler| self.row_rulers.insert_built(build.tab_row, Arc::new(ruler)));
        if inserted || self.ruler_requests_deferred {
            self.ruler_requests_deferred = false;
            cx.notify();
        }
    }
}

fn is_grid_byte(byte: u8) -> bool {
    (byte >= 0x20 && byte != 0x7f) || byte == b'\t' || byte == b'\n'
}

fn all_grid_bytes(bytes: &[u8]) -> bool {
    let found_control = bytes.iter().fold(false, |found_control, byte| {
        found_control | !is_grid_byte(*byte)
    });
    !found_control
}

fn map_row_through_tab_edits(row: u32, len: u32, patches: &[Vec<TabEdit>]) -> MappedRow {
    let mut row = row;
    let mut len = len;
    let mut kept = true;
    let mut interpolatable = true;
    let mut row_edits = Vec::new();
    for edits in patches {
        let row_start = TabPoint::new(row, 0);
        let row_end = TabPoint::new(row, len);
        let old_row = row;
        let mut patch_edits = Vec::new();
        let mut collect = |edit: &TabEdit| {
            if edit.old.start.row() == old_row
                && edit.old.end.row() == old_row
                && edit.new.start.row() == edit.new.end.row()
            {
                patch_edits.push(RowEdit {
                    old: edit.old.start.column()..edit.old.end.column(),
                    new_len: edit.new.end.column() - edit.new.start.column(),
                });
            } else {
                interpolatable = false;
            }
        };
        let ix = edits.partition_point(|edit| edit.old.end <= row_start);
        if let Some(edit) = ix.checked_sub(1).and_then(|ix| edits.get(ix)) {
            if edit.old.end == row_start && edit.new.end.column() != 0 {
                kept = false;
                collect(edit);
            }
            row = row - edit.old.end.row() + edit.new.end.row();
        }
        let mut overlapping = edits[ix..]
            .iter()
            .take_while(|edit| edit.old.start <= row_end)
            .peekable();
        if let Some(edit) = overlapping.peek() {
            kept = false;
            row = if edit.old.start.row() == old_row {
                edit.new.start.row()
            } else {
                edit.new.end.row()
            };
        }
        overlapping.for_each(&mut collect);
        let delta = patch_edits.iter().map(RowEdit::delta).sum::<i64>();
        len = u32::try_from(i64::from(len) + delta).unwrap_or(0);
        if !patch_edits.is_empty() {
            row_edits.push(patch_edits);
        }
    }
    if kept {
        MappedRow::Unchanged(row)
    } else if interpolatable {
        MappedRow::Edited {
            row,
            len,
            edits: row_edits,
        }
    } else {
        MappedRow::Changed(row)
    }
}

fn shift_trailing_whitespace_start(
    start: usize,
    patches: &[Vec<RowEdit>],
) -> Option<(usize, Option<Range<u32>>)> {
    let mut start = u32::try_from(start).ok()?;
    let mut inserted = None::<Range<u32>>;
    for edits in patches {
        let mut shifted_start = i64::from(start);
        let mut shifted_inserted = inserted.take().map(|range| {
            shift_column(range.start, edits, Bias::Left)
                ..shift_column(range.end, edits, Bias::Right)
        });
        let mut delta = 0i64;
        for edit in edits {
            if edit.old.end < start {
                shifted_start += edit.delta();
            } else if edit.old.start < start {
                return None;
            } else if edit.new_len > 0 {
                let edit_start = u32::try_from(i64::from(edit.old.start) + delta).ok()?;
                let edit_range = edit_start..edit_start + edit.new_len;
                shifted_inserted = Some(match shifted_inserted {
                    Some(range) => range.start.min(edit_range.start)..range.end.max(edit_range.end),
                    None => edit_range,
                });
            }
            delta += edit.delta();
        }
        start = u32::try_from(shifted_start).ok()?;
        inserted = shifted_inserted.filter(|range| !range.is_empty());
    }
    Some((start as usize, inserted))
}

fn shift_column(column: u32, edits: &[RowEdit], bias: Bias) -> u32 {
    let mut delta = 0i64;
    for edit in edits {
        if edit.old.start >= column {
            break;
        }
        if edit.old.end > column {
            let edit_start = i64::from(edit.old.start) + delta;
            let shifted = match bias {
                Bias::Left => edit_start,
                Bias::Right => edit_start + i64::from(edit.new_len),
            };
            return u32::try_from(shifted).unwrap_or(0);
        }
        delta += edit.delta();
    }
    u32::try_from(i64::from(column) + delta).unwrap_or(0)
}

fn last_non_whitespace_end(
    tab_snapshot: &TabSnapshot,
    row: u32,
    columns: Range<u32>,
) -> Option<usize> {
    let mut column = columns.start as usize;
    let mut end = None;
    for chunk in tab_snapshot.chunks(
        TabPoint::new(row, columns.start)..TabPoint::new(row, columns.end),
        LanguageAwareStyling {
            tree_sitter: false,
            diagnostics: false,
        },
        Highlights::default(),
    ) {
        if let Some((index, character)) = chunk
            .text
            .char_indices()
            .rfind(|(_, character)| !character.is_whitespace())
        {
            end = Some(column + index + character.len_utf8());
        }
        column += chunk.text.len();
    }
    end
}

fn anchor_hull(ranges: &[Range<Anchor>], buffer: &MultiBufferSnapshot) -> Option<Range<Anchor>> {
    let start = ranges
        .iter()
        .map(|range| range.start)
        .min_by(|a, b| a.cmp(b, buffer))?;
    let end = ranges
        .iter()
        .map(|range| range.end)
        .max_by(|a, b| a.cmp(b, buffer))?;
    Some(start..end)
}

fn tab_row_for_offset(fold_snapshot: &FoldSnapshot, offset: MultiBufferOffset) -> u32 {
    let inlay_snapshot = &fold_snapshot.inlay_snapshot;
    let inlay_point = inlay_snapshot.to_point(inlay_snapshot.to_inlay_offset(offset));
    fold_snapshot.to_fold_point(inlay_point, Bias::Left).row()
}

pub(super) fn row_buffer_range(
    fold_snapshot: &FoldSnapshot,
    tab_row: u32,
) -> Range<MultiBufferOffset> {
    let inlay_snapshot = &fold_snapshot.inlay_snapshot;
    let to_buffer = |fold_point: FoldPoint| {
        let inlay_point = fold_point.to_inlay_point(fold_snapshot);
        inlay_snapshot.to_buffer_offset(inlay_snapshot.to_offset(inlay_point))
    };
    to_buffer(FoldPoint::new(tab_row, 0))
        ..to_buffer(FoldPoint::new(tab_row, fold_snapshot.line_len(tab_row)))
}

fn uniform_ascii_key(font_id: FontId, font_size: Pixels, cell_width: Pixels) -> u64 {
    let mut hasher = FxHasher::default();
    font_id.hash(&mut hasher);
    hasher.write_u32(f32::from(font_size).to_bits());
    hasher.write_u32(f32::from(cell_width).to_bits());
    hasher.finish()
}

fn has_uniform_ascii_advances(
    text_system: &WindowTextSystem,
    font: &Font,
    font_size: Pixels,
    cell_width: Pixels,
) -> bool {
    let font_id = text_system.resolve_font(font);
    (0x20u8..=0x7E)
        .all(|byte| text_system.layout_width(font_id, font_size, byte as char) == cell_width)
}

fn column_from_usize(column: usize) -> u32 {
    u32::try_from(column).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FoldPlaceholder, MultiBuffer};
    use gpui::{App, Entity, TestAppContext, px};
    use itertools::Itertools as _;
    use language::LanguageAwareStyling;
    use project::project_settings::DiagnosticSeverity;
    use settings::SettingsStore;

    #[gpui::test]
    async fn test_ruler_builds_follow_requested_rows(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            crate::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let row_text = "漢字".repeat(3_000);
        assert!(row_text.len() as u32 > MAX_SYNC_RULER_LEN);
        let text = [row_text.as_str(); 4].join("\n");
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let map = cx.new(|cx| {
            DisplayMap::new(
                buffer,
                gpui::font("Courier"),
                px(14.),
                None,
                1,
                1,
                FoldPlaceholder::test(),
                DiagnosticSeverity::Warning,
                cx,
            )
        });
        let shaper = cx.update(|cx| RulerShaper {
            text_system: Arc::new(WindowTextSystem::new(cx.text_system().clone())),
            style: EditorStyle::default(),
            font_size: px(14.),
            language_aware: LanguageAwareStyling {
                tree_sitter: false,
                diagnostics: true,
            },
        });
        let render_rows = |rows: &[u32], focus: Option<u32>, cx: &mut TestAppContext| {
            map.update(cx, |map, cx| {
                let snapshot = map.snapshot(cx);
                for row in rows {
                    snapshot.ruled_row(DisplayRow(*row), shaper.clone());
                }
                map.spawn_ruler_builds(focus.map(DisplayRow), cx);
                map.ruler_builds
                    .iter()
                    .filter(|build| !build.is_cancelled())
                    .map(|build| build.tab_row)
                    .sorted()
                    .collect::<Vec<_>>()
            })
        };

        assert_eq!(render_rows(&[0, 1, 2, 3], Some(3), cx), [2, 3]);
        assert_eq!(render_rows(&[0, 2], None, cx), [2]);
        cx.run_until_parked();
        assert_eq!(render_rows(&[0], None, cx), [0]);
        cx.run_until_parked();
        let built = map.update(cx, |map, cx| {
            map.snapshot(cx)
                .ruled_row(DisplayRow(0), shaper.clone())
                .is_some_and(|ruled| ruled.exact)
        });
        assert!(built);
    }

    #[gpui::test]
    async fn test_replaced_rows_do_not_reuse_stale_rulers_as_layout(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            crate::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        for marker_edits in [0, MAX_PENDING_TAB_EDITS] {
            let markers = "\nx".repeat(marker_edits);
            let old_text = "漢".repeat(6_000) + &markers;
            let new_text = "é".repeat(9_000);
            let row_len = new_text.len() as u32;
            let buffer = cx.update(|cx| MultiBuffer::build_simple(&old_text, cx));
            let map = cx.new(|cx| {
                DisplayMap::new(
                    buffer.clone(),
                    gpui::font("Courier"),
                    px(14.),
                    None,
                    1,
                    1,
                    FoldPlaceholder::test(),
                    DiagnosticSeverity::Warning,
                    cx,
                )
            });
            let shaper = cx.update(|cx| RulerShaper {
                text_system: Arc::new(WindowTextSystem::new(cx.text_system().clone())),
                style: EditorStyle::default(),
                font_size: px(14.),
                language_aware: LanguageAwareStyling {
                    tree_sitter: false,
                    diagnostics: true,
                },
            });
            let render_row = |cx: &mut TestAppContext| {
                map.update(cx, |map, cx| {
                    let ruled = map
                        .snapshot(cx)
                        .ruled_row(DisplayRow(0), shaper.clone())
                        .map(|ruled| (ruled.exact, ruled.ruler.len()));
                    map.spawn_ruler_builds(Some(DisplayRow(0)), cx);
                    ruled
                })
            };

            assert_eq!(render_row(cx), None);
            cx.run_until_parked();
            assert_eq!(render_row(cx), Some((true, row_len)));

            let marker_offsets = old_text.match_indices('x').map(|(offset, _)| offset);
            let edits = std::iter::once((0..new_text.len(), new_text.as_str()))
                .chain(marker_offsets.map(|offset| (offset..offset + 1, "y")))
                .map(|(range, text)| {
                    (
                        MultiBufferOffset(range.start)..MultiBufferOffset(range.end),
                        text,
                    )
                })
                .collect::<Vec<_>>();
            buffer.update(cx, |buffer, cx| buffer.edit(edits, None, cx));
            assert_eq!(render_row(cx), None, "{marker_edits} marker edits");
            cx.run_until_parked();
            assert_eq!(render_row(cx), Some((true, row_len)));
        }
    }

    #[gpui::test]
    async fn test_estimated_windows_extend_until_they_cover_the_viewport(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings = SettingsStore::test(cx);
            cx.set_global(settings);
            crate::init(cx);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });
        let text = "e\u{301}".repeat(30_000);
        let buffer = cx.update(|cx| MultiBuffer::build_simple(&text, cx));
        let map = cx.new(|cx| {
            DisplayMap::new(
                buffer,
                gpui::font("Courier"),
                px(14.),
                None,
                1,
                1,
                FoldPlaceholder::test(),
                DiagnosticSeverity::Warning,
                cx,
            )
        });
        let snapshot = map.update(cx, |map, cx| map.snapshot(cx));
        let cell = GridCell {
            width: px(10.),
            monospace: false,
        };
        let viewport = HorizontalViewport {
            scroll_columns: 20_000.,
            visible_columns: 100.,
            text_align: TextAlign::Left,
            content_width: px(1_000.),
        };
        let row = DisplayRow(0);
        let estimated = EstimatedRow::new(&snapshot, row, snapshot.line_len(row));
        let initial = snapshot.estimated_window(&estimated, &viewport, cell);
        let render_width = |columns: &Range<u32>| px(columns.len() as f32 * 10. / 6.);
        let mut shapes = 0;
        let (covering, ()) =
            snapshot.covering_estimated_window(&estimated, &viewport, cell, |columns| {
                shapes += 1;
                ((), render_width(&columns))
            });
        let bound = snapshot.estimated_window_bound(&estimated, &viewport, cell);

        assert_eq!(covering.start, initial.start);
        assert_eq!(shapes, 2);
        assert!(covering.end > initial.end);
        let start_x = 10. * estimated.cells_before(covering.start) as ScrollPixelOffset;
        let visible_end = (viewport.scroll_columns + viewport.visible_columns) * 10.;
        assert!(start_x + ScrollPixelOffset::from(render_width(&covering)) >= visible_end);
        assert_eq!(bound.start, covering.start);
        assert!(bound.end >= covering.end);
    }

    #[gpui::test]
    fn test_control_rows_follow_edits_without_rescanning(cx: &mut App) {
        let buffer = MultiBuffer::build_simple("ab\u{1b}cd\nwxyz", cx);
        let offsets =
            |range: Range<usize>| MultiBufferOffset(range.start)..MultiBufferOffset(range.end);
        let edit = |buffer: &Entity<MultiBuffer>, range: Range<usize>, text: &str, cx: &mut App| {
            buffer.update(cx, |buffer, cx| {
                buffer.edit([(offsets(range), text)], None, cx)
            });
        };
        let tracked = |rows: &ControlRows, start: usize| {
            rows.0
                .lock()
                .get(&MultiBufferOffset(start))
                .map(|row| row.control.is_some())
        };

        let rows = ControlRows::default();
        let snapshot = buffer.read(cx).snapshot(cx);
        assert!(rows.contains_control(&snapshot, offsets(0..5)));
        assert!(!rows.contains_control(&snapshot, offsets(6..10)));

        edit(&buffer, 0..0, "q", cx);
        edit(&buffer, 8..8, "\u{7}", cx);
        let rows = rows.edited(
            &buffer.read(cx).snapshot(cx),
            vec![offsets(0..1), offsets(8..9)],
        );
        assert_eq!(
            (tracked(&rows, 0), tracked(&rows, 7)),
            (Some(true), Some(true))
        );

        edit(&buffer, 3..4, "", cx);
        let snapshot = buffer.read(cx).snapshot(cx);
        let rows = rows.edited(&snapshot, vec![offsets(3..3)]);
        assert_eq!(tracked(&rows, 0), Some(false));
        assert!(!rows.contains_control(&snapshot, offsets(0..5)));

        edit(&buffer, 5..5, "\n", cx);
        let rows = rows.edited(&buffer.read(cx).snapshot(cx), vec![offsets(5..6)]);
        assert_eq!(
            (tracked(&rows, 0), tracked(&rows, 7)),
            (Some(false), Some(true))
        );
        edit(&buffer, 5..7, "", cx);
        let snapshot = buffer.read(cx).snapshot(cx);
        let rows = rows.edited(&snapshot, vec![offsets(5..5)]);
        assert_eq!(tracked(&rows, 0), Some(true));
        assert!(rows.contains_control(&snapshot, offsets(0..10)));
    }
}
