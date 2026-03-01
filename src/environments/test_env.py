from foragers import ForagersEnv

env = ForagersEnv()

state, _ = env.reset()
print("Initial:", state)

for step in range(5):
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)
    print(f"Step {step} -> state={state}, reward={reward}")