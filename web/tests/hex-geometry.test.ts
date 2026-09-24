import assert from "node:assert/strict";
import test from "node:test";
import { hexBoardSize, hexPosition } from "../app/hex-geometry";

function axialDistance(firstQ: number, firstR: number, secondQ: number, secondR: number) {
  const dq = firstQ - secondQ;
  const dr = firstR - secondR;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

test("rendered edge adjacency matches the Rust axial topology across every column", () => {
  const cells = Array.from({ length: 11 * 9 }, (_, id) => ({ q: id % 11, r: Math.floor(id / 11) }));
  for (const [firstIndex, first] of cells.entries()) {
    const firstPosition = hexPosition(first.q, first.r, 9);
    for (const second of cells.slice(firstIndex + 1)) {
      const secondPosition = hexPosition(second.q, second.r, 9);
      const separation = Math.hypot(
        firstPosition.left - secondPosition.left,
        firstPosition.top - secondPosition.top,
      );
      assert.equal(
        separation < 4.1,
        axialDistance(first.q, first.r, second.q, second.r) === 1,
        `${first.q},${first.r} to ${second.q},${second.r}`,
      );
    }
  }
});

test("board bounds contain the wide axial grid", () => {
  for (const [columns, rows] of [[11, 9], [19, 15], [5, 2]]) {
    const board = hexBoardSize(columns, rows);
    for (let q = 0; q < columns; q += 1) {
      for (let r = 0; r < rows; r += 1) {
        const position = hexPosition(q, r, rows);
        assert.ok(position.left >= 0 && position.left + 4.625 <= board.width);
        assert.ok(position.top >= 0 && position.top + 4 <= board.height);
      }
    }
  }
});

test("two-player board uses the available horizontal game space", () => {
  const board = hexBoardSize(11, 9);
  assert.ok(board.width / board.height > 1.5);
  assert.ok(hexPosition(0, 4, 9).left < hexPosition(10, 4, 9).left);
});
