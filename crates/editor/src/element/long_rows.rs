use std::{iter, ops::Range};

use gpui::{App, Entity, Hsla, Pixels, TextAlign, Window};
use language::LanguageAwareStyling;
use multi_buffer::Anchor;
use sum_tree::Bias;
use theme::Theme;
use util::debug_panic;

use super::{LineFragment, LineWithInvisibles};
use crate::{
    DisplayPoint, DisplayRow, Editor, EditorSnapshot, EditorStyle,
    display_map::{
        DisplaySnapshot, EstimatedRow, GridCell, HighlightedChunk, HorizontalViewport, RuledRow,
        RulerShaper, ToDisplayPoint, WindowedRowGeometry,
    },
    movement::language_aware_styling,
    scroll::{ScrollOffset, ScrollPixelOffset},
};

const RULER_WIDTH_TOLERANCE: ScrollPixelOffset = 0.1;

pub(super) fn visible_highlight_ranges(
    snapshot: &EditorSnapshot,
    rows: Range<DisplayRow>,
    visible_range: Range<Anchor>,
    row_windowing: Option<(HorizontalViewport, GridCell)>,
    style: &EditorStyle,
    window: &Window,
) -> Vec<Range<Anchor>> {
    let Some((viewport, cell)) = row_windowing else {
        return vec![visible_range];
    };
    let buffer = snapshot.buffer_snapshot();
    let anchor_at =
        |point: DisplayPoint, bias: Bias| buffer.anchor_at(point.to_offset(snapshot, bias), bias);
    let mut ranges = Vec::new();
    let mut run_start = Some(visible_range.start);
    for row in rows.start.0..rows.end.0 {
        let display_row = DisplayRow(row);
        if let Some(row_len) = snapshot.long_unwrapped_row_len(display_row) {
            if let Some(run_start) = run_start.take() {
                ranges.push(run_start..anchor_at(DisplayPoint::new(display_row, 0), Bias::Right));
            }
            let bounds = long_row_columns(
                snapshot,
                display_row,
                row_len,
                &viewport,
                cell,
                style,
                window,
            );
            ranges.push(
                anchor_at(DisplayPoint::new(display_row, bounds.start), Bias::Left)
                    ..anchor_at(DisplayPoint::new(display_row, bounds.end), Bias::Right),
            );
            run_start = Some(anchor_at(
                DisplayPoint::new(display_row, row_len),
                Bias::Left,
            ));
        }
    }
    if let Some(run_start) = run_start {
        ranges.push(run_start..visible_range.end);
    }
    ranges.retain(|range| range.start.cmp(&range.end, buffer).is_lt());
    ranges
}

pub(super) fn background_highlights_in_ranges(
    editor: &Editor,
    ranges: &[Range<Anchor>],
    snapshot: &DisplaySnapshot,
    theme: &Theme,
) -> Vec<(Range<DisplayPoint>, Hsla)> {
    let mut highlights = Vec::new();
    let mut previous_end = None::<DisplayPoint>;
    for range in ranges {
        highlights.extend(
            editor
                .background_highlights_in_range(range.clone(), snapshot, theme)
                .into_iter()
                .filter(|(highlight, _)| previous_end.is_none_or(|end| highlight.start >= end)),
        );
        previous_end = Some(range.end.to_display_point(snapshot));
    }
    highlights
}

pub(super) fn layout_long_row(
    display_row: DisplayRow,
    row_len: u32,
    viewport: Option<HorizontalViewport>,
    cell: GridCell,
    snapshot: &EditorSnapshot,
    style: &EditorStyle,
    editor_width: Pixels,
    row_bg: &[Vec<(Range<DisplayPoint>, Hsla)>],
    window: &mut Window,
    cx: &mut App,
) -> LineWithInvisibles {
    let font_size = style.text.font_size.to_pixels(window.rem_size());
    let shaper = ruler_shaper(snapshot, display_row, style, window);
    let Some(viewport) = viewport else {
        let geometry = if snapshot.row_has_exact_grid(display_row, cell) {
            WindowedRowGeometry::new(row_len, cell, 0..0)
        } else {
            match ruled_row(snapshot, display_row, shaper) {
                Some(ruled) => WindowedRowGeometry::ruled(ruled, row_len, cell, 0..0),
                None => WindowedRowGeometry::estimated(
                    EstimatedRow::new(snapshot, display_row, row_len),
                    cell,
                    0..0,
                ),
            }
        };
        return LineWithInvisibles::empty(font_size).windowed(geometry);
    };

    let language_aware = shaper.language_aware;
    let shape = |chunks: &mut dyn Iterator<Item = HighlightedChunk<'_>>,
                 start: u32,
                 row_bg: &[Vec<(Range<DisplayPoint>, Hsla)>],
                 shape_without_backgrounds: bool,
                 window: &mut Window,
                 cx: &mut App| {
        LineWithInvisibles::from_chunks(
            chunks,
            style,
            usize::MAX,
            1,
            &snapshot.mode,
            editor_width,
            start as usize,
            |_| false,
            row_bg,
            shape_without_backgrounds,
            window,
            cx,
        )
        .pop()
        .unwrap_or_else(|| {
            debug_panic!("from_chunks always yields at least one layout");
            LineWithInvisibles::empty(font_size)
        })
    };
    let read = |bytes: Range<u32>| {
        snapshot.highlighted_chunks_in_range(
            DisplayPoint::new(display_row, bytes.start)..DisplayPoint::new(display_row, bytes.end),
            language_aware,
            style,
        )
    };
    let mut shaped;
    let geometry;
    if let Some((columns, _)) = snapshot.grid_window(display_row, row_len, &viewport, cell, &shaper)
    {
        shaped = shape(
            &mut read(columns.clone()),
            columns.start,
            row_bg,
            false,
            window,
            cx,
        );
        geometry = WindowedRowGeometry::new(row_len, cell, columns);
    } else if let Some(ruled) = ruled_row(snapshot, display_row, shaper) {
        let columns = ruled.columns_for_viewport(&viewport, cell);
        let pieces = ruled
            .ruler
            .render_pieces(columns.clone())
            .collect::<Vec<_>>();
        let read_range = pieces
            .first()
            .map_or(columns.start, |piece| piece.context.start)
            ..pieces.last().map_or(columns.end, |piece| piece.context.end);
        let chunks = read(read_range.clone()).collect::<Vec<_>>();
        let last = pieces.len().saturating_sub(1);
        let window_start_x = ruled.ruler.x_range_for_columns(columns.clone()).start;
        shaped = shape(&mut iter::empty(), columns.start, &[], false, window, cx);
        for (ix, piece) in pieces.into_iter().enumerate() {
            let mut part = shape(
                &mut clip_chunks(&chunks, read_range.start, piece.context.clone()),
                piece.context.start,
                row_bg,
                true,
                window,
                cx,
            );
            debug_assert!(
                !ruled.exact
                    || rendered_text_width(&part, &piece.context, &piece.chunk).is_none_or(
                        |width| {
                            (width - ScrollPixelOffset::from(piece.width)).abs()
                                <= RULER_WIDTH_TOLERANCE
                        }
                    ),
                "row {display_row:?} columns {:?} render {:?} px wide, the ruler expects {:?}",
                piece.chunk,
                rendered_text_width(&part, &piece.context, &piece.chunk),
                piece.width,
            );
            if piece.context != piece.chunk {
                part.extend_clipped_tabs(&snapshot.display_snapshot, display_row, &piece.context);
                part.trim_to(
                    &piece.context,
                    &piece.chunk,
                    piece.width,
                    (ix == 0, ix == last),
                );
            }
            shaped.append_at(part, Pixels::from(piece.x - window_start_x));
        }
        geometry = WindowedRowGeometry::ruled(ruled, row_len, cell, columns);
    } else {
        let estimated = EstimatedRow::new(snapshot, display_row, row_len);
        let (columns, line) =
            snapshot.covering_estimated_window(&estimated, &viewport, cell, |columns| {
                let line = shape(
                    &mut read(columns.clone()),
                    columns.start,
                    row_bg,
                    false,
                    window,
                    cx,
                );
                let width = line.shaped_width();
                (line, width)
            });
        shaped = line;
        geometry = WindowedRowGeometry::estimated(estimated, cell, columns);
    }

    shaped.extend_clipped_tabs(&snapshot.display_snapshot, display_row, geometry.window());
    if snapshot.shows_trailing_whitespace(display_row, cx) {
        let display_snapshot = &snapshot.display_snapshot;
        shaped.trailing_whitespace_start = Some(
            display_snapshot.cached_trailing_whitespace_start(display_row, || {
                trailing_whitespace_start(display_snapshot, display_row, row_len, style)
            }),
        );
    }
    shaped.windowed(geometry)
}

pub(super) fn anchored_scroll_x(
    editor: &Entity<Editor>,
    snapshot: &EditorSnapshot,
    rows: Range<DisplayRow>,
    viewport: HorizontalViewport,
    cell: GridCell,
    em_layout_width: Pixels,
    style: &EditorStyle,
    editor_width: Pixels,
    content_width: Pixels,
    window: &mut Window,
    cx: &mut App,
) -> Option<ScrollOffset> {
    let (anchor, offset) = editor.update(cx, |editor, cx| {
        editor.scroll_manager.take_horizontal_anchor(cx)
    })?;
    let em_layout_width = ScrollPixelOffset::from(em_layout_width);
    if em_layout_width <= 0. {
        return None;
    }
    let point = anchor.to_display_point(snapshot);
    if !rows.contains(&point.row()) {
        return None;
    }
    let row_len = snapshot.long_unwrapped_row_len(point.row())?;
    let shaper = ruler_shaper(snapshot, point.row(), style, window);
    if !ruled_row(snapshot, point.row(), shaper).is_some_and(|ruled| ruled.exact) {
        return None;
    }
    let layout = layout_long_row(
        point.row(),
        row_len,
        Some(viewport),
        cell,
        snapshot,
        style,
        editor_width,
        &[],
        window,
        cx,
    );
    let x = layout.alignment_offset(style.text.text_align, content_width)
        + layout.x_for_index(point.column() as usize);
    let scroll_x = ((x - offset) / em_layout_width).max(0.);
    if ((scroll_x - viewport.scroll_columns) * em_layout_width).abs() < 0.5 {
        return None;
    }
    editor.update(cx, |editor, cx| {
        let mut position = editor.scroll_position(cx);
        position.x = scroll_x;
        editor.set_scroll_position_internal(position, true, true, window, cx);
    });
    Some(scroll_x)
}

pub(super) fn provisional_horizontal_anchor(
    snapshot: &EditorSnapshot,
    start_row: DisplayRow,
    line_layouts: &[LineWithInvisibles],
    newest_head: DisplayPoint,
    scroll_left: ScrollPixelOffset,
    text_align: TextAlign,
    content_width: Pixels,
) -> Option<(Anchor, ScrollPixelOffset)> {
    let visible = scroll_left..=scroll_left + ScrollPixelOffset::from(content_width);
    let aligned_x = |layout: &LineWithInvisibles, column: usize| {
        layout.alignment_offset(text_align, content_width) + layout.x_for_index(column)
    };
    let provisional_layout = |row: DisplayRow| {
        row.0
            .checked_sub(start_row.0)
            .and_then(|ix| line_layouts.get(ix as usize))
            .filter(|layout| layout.is_provisional())
    };
    let (point, x) = provisional_layout(newest_head.row())
        .map(|layout| aligned_x(layout, newest_head.column() as usize))
        .filter(|x| visible.contains(x))
        .map(|x| (newest_head, x))
        .or_else(|| {
            line_layouts
                .iter()
                .enumerate()
                .filter(|(_, layout)| layout.is_provisional())
                .find_map(|(ix, layout)| {
                    let alignment = layout.alignment_offset(text_align, content_width);
                    let column = layout.index_for_x(scroll_left - alignment)?;
                    let row = DisplayRow(start_row.0 + ix as u32);
                    Some((
                        DisplayPoint::new(row, column as u32),
                        aligned_x(layout, column),
                    ))
                })
        })?;
    let anchor = snapshot
        .buffer_snapshot()
        .anchor_before(point.to_offset(snapshot, Bias::Left));
    Some((anchor, x - scroll_left))
}

pub(super) fn row_windowing(
    snapshot: &EditorSnapshot,
    rows: Range<DisplayRow>,
    viewport: HorizontalViewport,
    style: &EditorStyle,
    font_size: Pixels,
    window: &Window,
) -> Option<(HorizontalViewport, GridCell)> {
    (rows.start.0..rows.end.0)
        .any(|row| snapshot.is_long_unwrapped_row(DisplayRow(row)))
        .then(|| {
            let cell = GridCell::measure(window.text_system(), style, snapshot, font_size);
            (viewport, cell)
        })
}

fn long_row_columns(
    snapshot: &EditorSnapshot,
    display_row: DisplayRow,
    row_len: u32,
    viewport: &HorizontalViewport,
    cell: GridCell,
    style: &EditorStyle,
    window: &Window,
) -> Range<u32> {
    let shaper = ruler_shaper(snapshot, display_row, style, window);
    match snapshot.grid_window(display_row, row_len, viewport, cell, &shaper) {
        Some((columns, _)) => columns,
        None => match ruled_row(snapshot, display_row, shaper) {
            Some(ruled) => ruled.columns_for_viewport(viewport, cell),
            None => snapshot.estimated_window_bound(
                &EstimatedRow::new(snapshot, display_row, row_len),
                viewport,
                cell,
            ),
        },
    }
}

fn ruled_row(
    snapshot: &EditorSnapshot,
    display_row: DisplayRow,
    shaper: RulerShaper,
) -> Option<RuledRow> {
    if snapshot.mode.is_minimap() {
        None
    } else {
        snapshot.ruled_row(display_row, shaper)
    }
}

fn ruler_shaper(
    snapshot: &EditorSnapshot,
    display_row: DisplayRow,
    style: &EditorStyle,
    window: &Window,
) -> RulerShaper {
    RulerShaper {
        text_system: window.text_system().clone(),
        style: style.clone(),
        font_size: style.text.font_size.to_pixels(window.rem_size()),
        language_aware: language_aware_styling(
            snapshot,
            display_row,
            snapshot.semantic_tokens_enabled,
        ),
    }
}

fn clip_chunks<'a>(
    chunks: &'a [HighlightedChunk<'a>],
    chunks_start: u32,
    range: Range<u32>,
) -> impl Iterator<Item = HighlightedChunk<'a>> + 'a {
    let mut offset = chunks_start;
    chunks.iter().filter_map(move |chunk| {
        let chunk_range = offset..offset + chunk.text.len() as u32;
        offset = chunk_range.end;
        let start = chunk_range.start.max(range.start);
        let end = chunk_range.end.min(range.end);
        if start >= end {
            return None;
        }
        if chunk.replacement.is_some() && (start, end) != (chunk_range.start, chunk_range.end) {
            debug_panic!("shaping context must not split replaced chunks");
        }
        Some(HighlightedChunk {
            text: &chunk.text
                [(start - chunk_range.start) as usize..(end - chunk_range.start) as usize],
            style: chunk.style,
            diagnostic_underline_severity: chunk.diagnostic_underline_severity,
            is_tab: chunk.is_tab,
            is_inlay: chunk.is_inlay,
            replacement: chunk.replacement.clone(),
        })
    })
}

fn rendered_text_width(
    part: &LineWithInvisibles,
    context: &Range<u32>,
    chunk: &Range<u32>,
) -> Option<ScrollPixelOffset> {
    if part
        .fragments
        .iter()
        .any(|fragment| matches!(fragment, LineFragment::Element { .. }))
    {
        return None;
    }
    let x_at = |column: u32| part.x_for_index((column - context.start) as usize);
    Some(x_at(chunk.end) - x_at(chunk.start))
}

fn trailing_whitespace_start(
    snapshot: &DisplaySnapshot,
    row: DisplayRow,
    row_len: u32,
    style: &EditorStyle,
) -> usize {
    let language_aware = LanguageAwareStyling {
        tree_sitter: false,
        diagnostics: false,
    };
    let mut segment_end = row_len;
    let mut segment_len = 64;
    while segment_end > 0 {
        let clipped = snapshot.clip_ignoring_line_ends(
            DisplayPoint::new(row, segment_end.saturating_sub(segment_len)),
            Bias::Left,
        );
        let segment_start = if clipped.row() == row && clipped.column() < segment_end {
            clipped.column()
        } else {
            0
        };
        let mut column = segment_start as usize;
        let mut text_end = None;
        for chunk in snapshot.highlighted_chunks_in_range(
            DisplayPoint::new(row, segment_start)..DisplayPoint::new(row, segment_end),
            language_aware,
            style,
        ) {
            if let Some((index, character)) = chunk
                .text
                .char_indices()
                .rfind(|(_, character)| !character.is_whitespace())
            {
                text_end = Some(column + index + character.len_utf8());
            }
            column += chunk.text.len();
        }
        if let Some(text_end) = text_end {
            return text_end;
        }
        segment_end = segment_start;
        segment_len = segment_len.saturating_mul(2);
    }
    0
}
