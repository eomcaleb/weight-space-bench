"""Scores a network the way the prompt says: mean score per episode over N episodes of the official MinAtar 1.0.15
game, each action sampled from the softmax of the logits, sticky actions 0.1 (MinAtar's own wrapper), difficulty
ramping on, episodes cut at 10,000 steps. Seeds are fixed (episode i uses seed `seed + i` for the game and for the
action sampling), so every network is scored on the same episode stream.
"""
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

# The 6 input channels built from each game's own channels (the "THE INPUT" section of each prompt).
CHANNELS = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
}


def _env(game):
    import logging
    logging.getLogger().setLevel(logging.ERROR)          # MinAtar warns when matplotlib is missing (not needed)
    from minatar import Environment
    return Environment(game, sticky_action_prob=0.1, difficulty_ramping=True)


def _episodes(args):
    game, net, seeds, max_steps = args
    env = _env(game)
    src = [[env.env.channels[n] for n in names] for names in CHANNELS[game]]
    W1, b1, W2, b2, W3, b3 = (net[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    out = []
    for s in seeds:
        env.seed(int(s))
        env.reset()
        rng = np.random.default_rng([1, int(s)])
        total, steps, done = 0.0, 0, False
        while not done and steps < max_steps:
            st = env.state()
            x = np.zeros((10, 10, 6), bool)
            for ch, idx in enumerate(src):
                if idx:
                    x[:, :, ch] = st[:, :, idx].any(-1)
            nz = np.flatnonzero(x.reshape(-1))               # input i = (row * 10 + col) * 6 + channel
            h1 = np.tanh(W1[nz].sum(0) + b1)
            h2 = np.tanh(h1 @ W2 + b2)
            z = h2 @ W3 + b3
            p = np.exp(z - z.max())
            a = int(rng.choice(6, p=p / p.sum()))
            r, done = env.act(a)
            total += r
            steps += 1
        out.append((total, steps))
    return out


def score(net, game, episodes=1000, seed=0, max_steps=10000, workers=0):
    """-> {"mean", "se", "episodes", "max", "mean_length"} (the fields of the paper's minatar_eval.json)."""
    seeds = list(range(seed, seed + episodes))
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    chunks = [seeds[i::workers] for i in range(workers) if seeds[i::workers]]
    net = {k: np.asarray(v) for k, v in net.items()}
    res = []
    if len(chunks) == 1:
        res = _episodes((game, net, chunks[0], max_steps))
    else:
        with ProcessPoolExecutor(len(chunks)) as ex:
            for part in ex.map(_episodes, [(game, net, c, max_steps) for c in chunks]):
                res += part
    r = np.array([x[0] for x in res])
    n = np.array([x[1] for x in res])
    return {"mean": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
            "episodes": len(r), "max": float(r.max()), "mean_length": float(n.mean())}
