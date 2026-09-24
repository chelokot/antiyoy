const HEX_WIDTH_REM = 4.625;
const HEX_HEIGHT_REM = 4;
const COLUMN_STEP_REM = HEX_WIDTH_REM * 0.75;

export function hexPosition(q: number, r: number) {
  return {
    left: q * COLUMN_STEP_REM,
    top: (r + q / 2) * HEX_HEIGHT_REM,
  };
}

export function hexBoardSize(columns: number, rows: number) {
  return {
    width: HEX_WIDTH_REM + (columns - 1) * COLUMN_STEP_REM,
    height: rows * HEX_HEIGHT_REM + (columns - 1) * HEX_HEIGHT_REM / 2,
  };
}
