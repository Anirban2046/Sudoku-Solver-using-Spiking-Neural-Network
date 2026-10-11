import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import App from "../App";

afterEach(cleanup);

const start = () => screen.getByRole("button", { name: "Start Solving" }) as HTMLButtonElement;
const cell = (r: number, c: number) => screen.getByLabelText(`row ${r} column ${c}`);
const type = (r: number, c: number, key: string) => fireEvent.keyDown(cell(r, c), { key });

describe("App", () => {
  it("shows exactly the title, the grid and the three buttons", () => {
    render(<App />);
    expect(screen.getByRole("heading", { level: 1 }).textContent).toBe("Sudoku Solver using Spiking Neural Network");
    expect(screen.getAllByRole("textbox")).toHaveLength(81);
    expect(screen.getAllByRole("button").map((b) => b.textContent)).toEqual([
      "Randomly Initialize",
      "Clear Grid",
      "Start Solving",
    ]);
  });

  it("clears the grid", () => {
    render(<App />);
    const clear = screen.getByRole("button", { name: "Clear Grid" }) as HTMLButtonElement;
    expect(clear.disabled).toBe(true);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    expect(clear.disabled).toBe(false);
    act(() => fireEvent.click(clear));
    const filled = (screen.getAllByRole("textbox") as HTMLInputElement[]).filter((el) => el.value !== "");
    expect(filled).toHaveLength(0);
    expect(clear.disabled).toBe(true);
  });

  it("keeps the other digits when a cell is edited after Randomly Initialize", () => {
    render(<App />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    const inputs = screen.getAllByRole("textbox") as HTMLInputElement[];
    const before = inputs.filter((el) => el.value).length;
    const empty = inputs.findIndex((el) => !el.value);
    act(() => fireEvent.keyDown(inputs[empty], { key: "1" }));
    expect((screen.getAllByRole("textbox") as HTMLInputElement[]).filter((el) => el.value).length).toBe(before + 1);
  });

  it("disables Start until there are 17 givens", () => {
    render(<App />);
    expect(start().disabled).toBe(true);
    expect(screen.getByText("Enter at least 17 numbers")).toBeTruthy();
  });

  it("enables Start after Randomly Initialize", () => {
    render(<App />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    const filled = screen.getAllByRole("textbox").filter((el) => (el as HTMLInputElement).value !== "");
    expect(filled.length).toBeGreaterThanOrEqual(36);
    expect(start().disabled).toBe(false);
  });

  it("flags duplicates in red and disables Start", () => {
    render(<App />);
    act(() => fireEvent.click(screen.getByRole("button", { name: "Randomly Initialize" })));
    // find a given digit and copy it into an empty cell of the same row
    const inputs = screen.getAllByRole("textbox") as HTMLInputElement[];
    let done = false;
    for (let r = 0; r < 9 && !done; r++) {
      const row = inputs.slice(r * 9, r * 9 + 9);
      const given = row.find((el) => el.value);
      const empty = row.findIndex((el) => !el.value);
      if (given && empty >= 0) {
        act(() => type(r + 1, empty + 1, given.value));
        done = true;
      }
    }
    expect(done).toBe(true);
    expect(start().disabled).toBe(true);
    expect(screen.getByText("Fix the duplicates shown in red")).toBeTruthy();
    expect(screen.getAllByText(/appears twice in row/).length).toBeGreaterThan(0);
    expect(document.querySelectorAll(".cell.conflict").length).toBeGreaterThanOrEqual(2);
  });

  it("disables Start for an unsolvable puzzle", () => {
    render(<App />);
    // (1,1) sees 1-4 in its row, 5, 6, 9 in its column and 7, 8 in its box: no candidates.
    const entries: [number, number, string][] = [
      [1, 2, "1"], [1, 3, "2"], [1, 4, "3"], [1, 5, "4"], [2, 1, "5"], [3, 1, "6"], [2, 2, "7"], [3, 3, "8"], [6, 1, "9"],
      // filler givens elsewhere, conflict-free
      [9, 9, "1"], [8, 8, "2"], [7, 7, "3"], [9, 5, "5"], [8, 4, "6"], [7, 6, "7"], [5, 9, "8"], [4, 8, "9"], [5, 5, "1"],
    ];
    for (const [r, c, k] of entries) act(() => type(r, c, k));
    expect(start().disabled).toBe(true);
    expect(screen.getByText("This puzzle has no solution")).toBeTruthy();
  });

  it("supports clearing and arrow navigation", () => {
    render(<App />);
    act(() => type(1, 1, "5"));
    expect((cell(1, 1) as HTMLInputElement).value).toBe("5");
    act(() => type(1, 1, "Backspace"));
    expect((cell(1, 1) as HTMLInputElement).value).toBe("");
    act(() => type(1, 1, "x"));
    expect((cell(1, 1) as HTMLInputElement).value).toBe("");
    cell(1, 1).focus();
    act(() => type(1, 1, "ArrowRight"));
    expect(document.activeElement).toBe(cell(1, 2));
  });
});
