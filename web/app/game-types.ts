export type CellView = {
  id: number;
  playable: boolean;
  owner: number | null;
  object: string;
  strength: number;
  ready: boolean;
  province: number | null;
  defense: number;
};

export type ProvinceView = {
  id: number;
  owner: number;
  money: number;
  income: number;
  upkeep: number;
  profit: number;
  capital: number;
  size: number;
};

export type CoreAction =
  | "EndTurn"
  | { Move: { source: number; target: number } }
  | { Recruit: { province: number; target: number; strength: number } }
  | { Build: { target: number; structure: string } }
  | { PlantTree: { target: number } }
  | { Diplomacy: { target: number; command: string } };

export type StateView = {
  width: number;
  height: number;
  round: number;
  active_player: number;
  terminal: boolean;
  winner: number | null;
  cells: CellView[];
  provinces: ProvinceView[];
  relations: Array<{
    first: number;
    second: number;
    relation: string;
    proposal: string | null;
  }>;
  legal_actions: CoreAction[];
};
