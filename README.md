# VALUE ITERATION ALGORITHM

## AIM

To implement and understand the Value Iteration Algorithm for solving a Markov Decision Process (MDP) and to compute the optimal policy and optimal value function for the given environment.

## PROBLEM STATEMENT

Reinforcement Learning problems can be modeled as Markov Decision Processes (MDPs), where an agent interacts with an environment by taking actions, receiving rewards, and transitioning between states.
The aim is to find the optimal policy π* that maximizes the expected cumulative reward. The Value Iteration Algorithm is a dynamic programming method that iteratively updates the value function until it converges, and then derives the optimal policy.
In this experiment, the problem is to apply Value Iteration to the given environment (e.g., FrozenLake-v1 or gym-walk environment) and compute:

The optimal value function V*

The optimal policy π*

The success rate of the policy through simulation.


## VALUE ITERATION ALGORITHM
Initialize the value function 
𝑉
(
𝑠
)
V(s) arbitrarily for all states 
𝑠
s.

Repeat until convergence (difference < θ):

For each state 
𝑠
s:
<img width="447" height="82" alt="image" src="https://github.com/user-attachments/assets/33dd20e2-5b41-4444-9661-c1fa04c0ccbb" />

Derive policy: For each state 
𝑠
s, choose the action 
𝑎
a that maximizes the expected return:

<img width="447" height="82" alt="image" src="https://github.com/user-attachments/assets/e0d0e1be-a300-4e42-a4e1-6c9098afb849" />

Output the optimal policy π* and optimal value function V*.
## VALUE ITERATION FUNCTION
### Name: BALASUBRAMANIAM
### Register Number:212224240020
```python
def value_iteration(P, gamma=1.0, theta=1e-10):
    V = np.zeros(len(P), dtype=np.float64)
    while True:
        Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
        for s in range(len(P)):
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
        if np.max(np.abs(V - np.max(Q, axis=1))) < theta:
            break
        V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return V, pi
```
## OUTPUT:
<img width="487" height="127" alt="image" src="https://github.com/user-attachments/assets/bb47504e-77c7-4ed9-9e78-bb70dd6f6256" />
<img width="702" height="39" alt="image" src="https://github.com/user-attachments/assets/34ce7502-efbe-419e-8a24-867f206acfb3" />
<img width="594" height="116" alt="image" src="https://github.com/user-attachments/assets/376869ab-a89f-45dc-8998-b3ffb9339727" />

## RESULT:

The Value Iteration Algorithm was successfully implemented. The computed optimal policy guides the agent to reach the goal state efficiently, and the optimal value function represents the maximum expected return for each state. The algorithm achieves a high success rate, validating the correctness of the derived policy.
