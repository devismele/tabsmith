import {
  buildScore,
  flagsForValue,
  type Score,
  type ScoreColumn,
  type ScoreMeasure,
} from "./score";
import type { AnalysisResult } from "./types";
import {
  columnXInBox,
  cursorAtTime,
  layoutScore,
  ROW_BOTTOM_PAD,
  ROW_TOP_PAD,
  STRING_BLOCK,
  STRING_GAP,
  stringY,
  visibleRowRange,
  type MeasureBox,
  type RowBox,
  type TabLayout,
} from "./tabLayout";

const NS = "http://www.w3.org/2000/svg";
const STRING_LABELS = ["e", "B", "G", "D", "A", "E"];
const STEM_TOP = ROW_TOP_PAD + STRING_BLOCK + 10;
const STEM_BOTTOM = STEM_TOP + 20;

type SvgAttrs = Record<string, string | number>;

function svg(tag: string, attrs: SvgAttrs = {}, text?: string): SVGElement {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

export type TabRendererOptions = {
  onSeek?: (time: number) => void;
  onSelectNote?: (noteIndex: number | null) => void;
};

/**
 * SVG tablature renderer driven by canonical score data. Only rows within (or
 * near) the viewport are drawn, so very long songs stay responsive. The
 * playhead is a cheap element moved every animation frame.
 */
export class TabRenderer {
  private readonly scroll: HTMLDivElement;
  private readonly svgRoot: SVGSVGElement;
  private readonly rowsLayer: SVGGElement;
  private readonly activeMeasureRect: SVGRectElement;
  private readonly playhead: SVGLineElement;
  private readonly resizeObserver: ResizeObserver;

  private score: Score | null = null;
  private layout: TabLayout | null = null;
  private zoom = 1;
  private countOverlay = false;
  private renderedRange = { first: 0, last: -1 };
  private time = 0;
  private selectedNoteIndex: number | null = null;
  private availableWidth = 0;

  constructor(private readonly container: HTMLElement, private readonly options: TabRendererOptions = {}) {
    container.classList.add("tabx");
    this.scroll = document.createElement("div");
    this.scroll.className = "tabx-scroll";
    this.svgRoot = svg("svg", { class: "tabx-svg" }) as SVGSVGElement;
    this.rowsLayer = svg("g", { class: "tabx-rows" }) as SVGGElement;
    this.activeMeasureRect = svg("rect", { class: "tabx-active-measure", x: 0, y: 0, width: 0, height: 0, rx: 6 }) as SVGRectElement;
    this.activeMeasureRect.style.display = "none";
    this.playhead = svg("line", { class: "tabx-playhead", x1: 0, y1: 0, x2: 0, y2: 0 }) as SVGLineElement;
    this.playhead.style.display = "none";
    this.svgRoot.append(this.activeMeasureRect, this.rowsLayer, this.playhead);
    this.scroll.append(this.svgRoot);
    container.append(this.scroll);

    this.scroll.addEventListener("scroll", () => this.renderVisible());
    this.svgRoot.addEventListener("click", (event) => this.handleClick(event));
    this.resizeObserver = new ResizeObserver(() => this.relayout());
    this.resizeObserver.observe(this.scroll);
  }

  setScore(analysis: Pick<AnalysisResult, "notes" | "duration" | "bpm" | "tuning" | "capo">): void {
    this.score = buildScore(analysis);
    this.selectedNoteIndex = null;
    this.relayout(true);
  }

  setZoom(zoom: number): void {
    this.zoom = Math.max(0.5, Math.min(2.4, zoom));
    this.relayout(true);
  }

  getZoom(): number {
    return this.zoom;
  }

  setCountOverlay(enabled: boolean): void {
    this.countOverlay = enabled;
    this.relayout(true);
  }

  private relayout(force = false): void {
    if (!this.score) return;
    const width = this.scroll.clientWidth || this.container.clientWidth;
    if (!force && width === this.availableWidth) return;
    this.availableWidth = width;
    this.layout = layoutScore(this.score, { availableWidth: width, zoom: this.zoom });
    this.svgRoot.setAttribute("width", String(this.layout.width));
    this.svgRoot.setAttribute("height", String(this.layout.height));
    this.svgRoot.setAttribute("viewBox", `0 0 ${this.layout.width} ${this.layout.height}`);
    this.renderedRange = { first: 0, last: -1 };
    this.renderVisible(true);
    this.setTime(this.time, false);
  }

  private renderVisible(force = false): void {
    if (!this.score || !this.layout) return;
    const range = visibleRowRange(this.layout, this.scroll.scrollTop, this.scroll.clientHeight);
    if (!force && range.first === this.renderedRange.first && range.last === this.renderedRange.last) return;
    this.renderedRange = range;
    this.rowsLayer.replaceChildren();
    for (let i = range.first; i <= range.last; i += 1) {
      this.rowsLayer.append(this.renderRow(this.layout.rows[i]));
    }
    this.paintActiveNotes();
  }

  private renderRow(row: RowBox): SVGGElement {
    const group = svg("g", { transform: `translate(0 ${row.y})` }) as SVGGElement;
    if (!this.score) return group;

    const first = row.measures[0];
    const last = row.measures[row.measures.length - 1];
    const left = first.x + 8;
    const right = last.x + last.width + 8;

    // Six string lines spanning the row, with pitch labels on the left.
    for (let string = 1; string <= 6; string += 1) {
      const y = stringY(string);
      group.append(svg("line", { class: "tabx-string", x1: left, y1: y, x2: right, y2: y }));
      group.append(svg("text", { class: "tabx-string-label", x: left - 6, y: y + 3, "text-anchor": "end" }, STRING_LABELS[string - 1]));
    }

    for (const box of row.measures) {
      this.renderMeasure(group, this.score.measures[box.measureIndex], box);
    }
    return group;
  }

  private renderMeasure(group: SVGGElement, measure: ScoreMeasure, box: MeasureBox): void {
    const barLeft = box.x + 8;
    const barRight = box.x + box.width + 8;
    const topY = stringY(1);
    const bottomY = stringY(6);

    // Bar lines and measure number.
    group.append(svg("line", { class: "tabx-barline", x1: barLeft, y1: topY, x2: barLeft, y2: bottomY }));
    group.append(svg("line", { class: "tabx-barline", x1: barRight, y1: topY, x2: barRight, y2: bottomY }));
    group.append(svg("text", { class: "tabx-measure-number", x: barLeft + 2, y: ROW_TOP_PAD - 12 }, String(measure.index + 1)));

    if (this.countOverlay) this.renderCountOverlay(group, box);

    // Transparent hit target for click-to-seek on the measure.
    const hit = svg("rect", {
      class: "tabx-measure-hit",
      x: barLeft,
      y: 0,
      width: Math.max(1, barRight - barLeft),
      height: ROW_TOP_PAD + STRING_BLOCK + ROW_BOTTOM_PAD,
      "data-measure-start": measure.start,
    });
    group.append(hit);

    for (const column of measure.columns) this.renderColumn(group, column, box);
  }

  private renderColumn(group: SVGGElement, column: ScoreColumn, box: MeasureBox): void {
    const x = columnXInBox(box, column.fraction);

    for (const note of column.notes) {
      const y = stringY(note.string);
      const label = String(note.fret);
      // Sustain line to the right, drawn under the number.
      const sustainFraction = Math.min(1, column.fraction + note.sustainSlots / 16);
      const sustainRight = columnXInBox(box, sustainFraction) - 6;
      if (note.sustainSlots > 1 && sustainRight > x + 8) {
        group.append(svg("line", { class: "tabx-sustain", x1: x + 8, y1: y, x2: sustainRight, y2: y }));
      }
      const width = 8 + label.length * 6;
      group.append(svg("rect", { class: "tabx-fret-bg", x: x - width / 2, y: y - 7, width, height: 14, rx: 3 }));
      const text = svg("text", {
        class: "tabx-fret",
        x,
        y: y + 4,
        "text-anchor": "middle",
        "data-note-index": note.noteIndex,
        "data-note-start": note.start,
        "data-note-end": note.end,
      }, label) as SVGTextElement;
      if (note.confidence < 0.4) text.classList.add("low-confidence");
      else if (note.confidence < 0.6) text.classList.add("medium-confidence");
      if (note.noteIndex === this.selectedNoteIndex) text.classList.add("selected");
      group.append(text);
    }

    // Rhythm stem below the strings.
    group.append(svg("line", { class: "tabx-stem", x1: x, y1: STEM_TOP, x2: x, y2: STEM_BOTTOM }));
    const flags = flagsForValue(column.value);
    for (let f = 0; f < flags; f += 1) {
      const fy = STEM_BOTTOM - f * 5;
      group.append(svg("line", { class: "tabx-flag", x1: x, y1: fy, x2: x + 8, y2: fy - 4 }));
    }
    if (column.dotted) group.append(svg("circle", { class: "tabx-dot", cx: x + 6, cy: STEM_BOTTOM, r: 1.6 }));
  }

  private renderCountOverlay(group: SVGGElement, box: MeasureBox): void {
    const labels = ["1", "e", "&", "a", "2", "e", "&", "a", "3", "e", "&", "a", "4", "e", "&", "a"];
    for (let slot = 0; slot < 16; slot += 1) {
      const x = columnXInBox(box, slot / 16);
      group.append(svg("text", { class: "tabx-count", x, y: ROW_TOP_PAD - 2, "text-anchor": "middle" }, labels[slot]));
    }
  }

  private handleClick(event: MouseEvent): void {
    const target = event.target as SVGElement;
    const noteEl = target.closest<SVGElement>("[data-note-index]");
    if (noteEl) {
      const noteIndex = Number(noteEl.getAttribute("data-note-index"));
      this.selectNote(noteIndex);
      this.options.onSeek?.(Number(noteEl.getAttribute("data-note-start")));
      return;
    }
    const measureEl = target.closest<SVGElement>("[data-measure-start]");
    if (measureEl) {
      this.selectNote(null);
      this.options.onSeek?.(Number(measureEl.getAttribute("data-measure-start")));
    }
  }

  selectNote(noteIndex: number | null): void {
    if (this.selectedNoteIndex === noteIndex) return;
    this.selectedNoteIndex = noteIndex;
    this.rowsLayer.querySelectorAll(".tabx-fret.selected").forEach((el) => el.classList.remove("selected"));
    if (noteIndex !== null) {
      this.rowsLayer.querySelector(`.tabx-fret[data-note-index="${noteIndex}"]`)?.classList.add("selected");
    }
    this.options.onSelectNote?.(noteIndex);
  }

  /** Moves the playhead and highlights. `follow` auto-scrolls to keep it visible. */
  setTime(time: number, follow = true): void {
    this.time = time;
    if (!this.score || !this.layout) return;
    const position = cursorAtTime(this.score, this.layout, time);
    if (!position) {
      this.playhead.style.display = "none";
      return;
    }
    const top = position.rowY + ROW_TOP_PAD - 20;
    const bottom = position.rowY + ROW_TOP_PAD + STRING_BLOCK + ROW_BOTTOM_PAD - 20;
    this.playhead.style.display = "";
    this.playhead.setAttribute("x1", String(position.x));
    this.playhead.setAttribute("x2", String(position.x));
    this.playhead.setAttribute("y1", String(top));
    this.playhead.setAttribute("y2", String(bottom));

    this.updateActiveMeasure(position.measureIndex);
    if (follow) this.followRow(position.rowY);
    this.paintActiveNotes();
  }

  private updateActiveMeasure(measureIndex: number): void {
    const located = this.layout?.measureLocation.get(measureIndex);
    if (!located) {
      this.activeMeasureRect.style.display = "none";
      return;
    }
    const { box, rowIndex } = located;
    const rowY = this.layout!.rows[rowIndex].y;
    this.activeMeasureRect.style.display = "";
    this.activeMeasureRect.setAttribute("x", String(box.x + 8));
    this.activeMeasureRect.setAttribute("y", String(rowY + ROW_TOP_PAD - 22));
    this.activeMeasureRect.setAttribute("width", String(box.width));
    this.activeMeasureRect.setAttribute("height", String(STRING_BLOCK + 30));
  }

  private followRow(rowY: number): void {
    const viewTop = this.scroll.scrollTop;
    const viewBottom = viewTop + this.scroll.clientHeight;
    const rowBottom = rowY + ROW_TOP_PAD + STRING_BLOCK + ROW_BOTTOM_PAD;
    const margin = STRING_GAP * 2;
    if (rowY < viewTop + margin) {
      this.scroll.scrollTo({ top: Math.max(0, rowY - margin), behavior: "smooth" });
    } else if (rowBottom > viewBottom - margin) {
      this.scroll.scrollTo({ top: rowBottom - this.scroll.clientHeight + margin, behavior: "smooth" });
    }
  }

  private paintActiveNotes(): void {
    const time = this.time;
    this.rowsLayer.querySelectorAll<SVGTextElement>(".tabx-fret").forEach((el) => {
      const start = Number(el.getAttribute("data-note-start"));
      const end = Number(el.getAttribute("data-note-end"));
      el.classList.toggle("live", time >= start && time < end);
    });
  }

  destroy(): void {
    this.resizeObserver.disconnect();
    this.container.classList.remove("tabx");
    this.scroll.remove();
  }
}
