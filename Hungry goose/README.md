# [Hungry Geese](https://www.kaggle.com/c/hungry-geese/overview)

## 우리가 일반적으로 했었던 그 스네이크 게임입니다. 단, 혼자하는게 아닌 다른 플레이어 3명과 함께 그리고 먹이가 단 2개만 제공됩니다.
## The snake game we played before, but not alone, with other 3 players and there is only 2 foods in map

### q_agent

- Q-learning을 활용한 agent
- Terminal : 먹이를 먹거나 죽었을 때
- 먹이에 대한 reward는 100, 죽었을 때의 reward는 -999
- 현재 맵 상황에서 BFS를 활용하여 먹이까지의 최소 거리를 얻고, 다른 뱀들과 비교했을 때 최소 거리가 아니라면 자신의 꼬리에 reward를 1점 준다.
- 상대의 머리가 이동할 수 있는 가능성이 있는 위치에는 -9점을 줘서 최대한의 충돌 가능성을 없앤다.

- Q-learning based agent
- Terminal : When snake eat the food or dead
- Reward for food is 100, for Dead is -999
- Find the min distance with BFS Algorithm in present map. If the distance is not minimum distance compared with other snakes, set reward for tail as 1
- The cells that have some of probability which other snakes move to is set reward as -9

### Result with greedy Agent
- White one is q_agent
![result of q learning agent](./q_agent_result/result.apng)
