const HEX_WIDTH_REM = 4.625;
const HEX_HEIGHT_REM = 4;
const COLUMN_STEP_REM = HEX_WIDTH_REM * 0.75;

export function hexPosition(q: number, r: number, rows: number) {
  return {
    left: (q + r) * COLUMN_STEP_REM,
    top: (rows - 1 + q - r) * HEX_HEIGHT_REM / 2,
  };
}

export function hexBoardSize(columns: number, rows: number) {
  return {
    width: HEX_WIDTH_REM + (columns + rows - 2) * COLUMN_STEP_REM,
    height: HEX_HEIGHT_REM + (columns + rows - 2) * HEX_HEIGHT_REM / 2,
  };
}
