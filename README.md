# Skull King Reinforcement Learning

## Overview
This project implements a reinforcement learning environment for the card game Skull King using the OpenAI Gym interface. The environment is designed to simulate the game mechanics, allowing agents to learn strategies for bidding and playing cards.

## Files
- **env/SKEnvNoSpecials.py**: Defines the `SkullKingEnvNoSpecials` class, which implements the Skull King game environment. It includes methods for creating the deck, dealing cards, resetting the environment, taking steps in the game, resolving tricks, calculating rewards, and transitioning to the next round.

- **agent/agent.py**: Contains the implementation of the agent that interacts with the `SkullKingEnvNoSpecials` environment. The agent will employ strategies for bidding and playing cards based on the current state of the game. The action space is dynamically adjusted to accommodate the varying number of cards each player has.

- **main.py**: The entry point for the project. This file initializes the environment and the agent, runs the game loop, and manages the interaction between the agent and the environment.

- **requirements.txt**: Lists the dependencies required for the project, including `gym` and `numpy`.

## Game Rules
1. **Dealing Phase**: 
   - Shuffle the deck.
   - Deal `round_number` cards to each player.
   - Initialize the bidding phase.

2. **Bidding Phase**: 
   - Each player chooses a bid (0 to `round_number`).
   - Store bids and transition to the trick-playing phase.

3. **Trick-Playing Phase**: 
   - Players play tricks following the Skull King rules.
   - Determine trick winners and track the number of tricks won.

4. **Scoring Phase**: 
   - Compute rewards based on bid accuracy.
   - Move to the next round.

## Setup Instructions
1. Clone the repository:
   ```
   git clone <repository-url>
   cd skull-king-rl
   ```

2. Install the required dependencies and make the project editable:
   ```
   pip install -r requirements.txt
   pip install -e .
   ```

3. Launch an experiment:
   ```
   python main.py --num_episodes 100 --num_players 3 --agent_types learning simple simple --seed 42 --debug
   ```

3. Run the main script to start the game:
   ```
   python main.py
   ```

## Running the Agent
To run the agent within the environment, ensure that the agent's logic is implemented in `agent/agent.py`. The agent will interact with the `SkullKingEnvNoSpecials` environment, making decisions based on the current game state.

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.