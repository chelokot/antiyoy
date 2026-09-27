type BotTurnState = {
  terminal: boolean;
  active_player: number;
};

type BotReply<State extends BotTurnState> = {
  state: State;
  actions: number;
};

function needsBotAction<State extends BotTurnState>(state: State, humanSeat: number): boolean {
  return !state.terminal && state.active_player !== humanSeat;
}

function takeBotAction<State extends BotTurnState>(
  reply: BotReply<State>,
  step: () => State,
): void {
  if (reply.actions >= 2_000) {
    throw new Error("Bot response exceeded 2000 actions");
  }
  reply.state = step();
  reply.actions += 1;
}

export function advanceBotReply<State extends BotTurnState>(
  initial: State,
  humanSeat: number,
  step: () => State,
): BotReply<State> {
  const reply = { state: initial, actions: 0 };
  while (needsBotAction(reply.state, humanSeat)) {
    takeBotAction(reply, step);
  }
  return reply;
}

export async function advanceBotReplyCooperatively<State extends BotTurnState>(
  initial: State,
  humanSeat: number,
  step: () => State,
  yieldControl: () => Promise<void>,
  isCurrent: () => boolean,
): Promise<BotReply<State> | null> {
  const reply = { state: initial, actions: 0 };
  while (needsBotAction(reply.state, humanSeat)) {
    await yieldControl();
    if (!isCurrent()) {
      return null;
    }
    takeBotAction(reply, step);
  }
  return reply;
}
