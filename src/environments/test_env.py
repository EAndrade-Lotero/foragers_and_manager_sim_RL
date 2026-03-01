from foragers import ForagersEnv

env = ForagersEnv(seed=0, max_turns=5, persistent_world=True)

state, info = env.reset()
print("reset:", state, info)

for t in range(10):
    action = env.action_space.sample()
    state, reward, done, truncated, info = env.step(action)
    print(
        f"t={t} action={float(action[0]):.3f} state={state} reward={reward:.3f} "
        f"done={done} harvest={info.get('harvest')} gini={info.get('gini'):.3f} instab={info.get('instability'):.3f}"
    )
    if done:
        break