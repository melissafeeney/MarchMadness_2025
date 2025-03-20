🏀 Can Machine Learning Crack the 2025 March Madness Bracket? 🤖🏀


March Madness is back, and once again, I’ve created a machine learning algorithm to generate my bracket. My approach combines two powerful techniques in deep learning: **autoencoders** and **residual connections**. I trained the model on 100K+ games (2003–2025 pre-tournament data) to identify key patterns.
<br>
<br>

⚙️ **Feature Engineering: Laying the Foundation**

Before modeling, I engineered features to capture meaningful signals in team performance:
 ✅ Strength of schedule proxies from KenPom, Massey, and RPI rankings
 ✅ Season-level stats (coach seniority, win/loss record, shooting efficiency, assist-to-turnover ratio)
 ✅ Rolling 10-game win percentage to track momentum shifts
 ✅ "Hot streaks"- performance in the final 10 regular-season games (win % and consecutive wins)

While these features provide valuable context, the real magic happens when autoencoders and residual connections refine and enhance this data for better predictions.
<br>
<br>

🔍 **Autoencoders: The Data Distillers**

Think of autoencoders as data filters that extract the most relevant insights from raw, noisy sports data—scores, player stats, ratios, etc. Here’s how they improve predictions:
 ✅ Compression & Clarity: Condenses massive datasets into a structured, meaningful representation.
 ✅ Feature Extraction: Refines the raw game data, highlighting the most important "signals" while filtering out noise.
 ✅ Better Predictions: Provides the predictor model with a cleaner, more focused input dataset.
<br>
<br>

🚀 **Residual Connections: The Information Highways**

Deep learning models often struggle as they grow more complex- residual connections solve this by allowing information to flow more efficiently:
 ✅ Shortcut Paths: Creates direct routes between non-adjacent layers, preventing valuable details from getting lost (think of the express route vs. the local route).
 ✅ Gradient Flow: Improves training efficiency by allowing information to pass through smoothly- reducing the risk of vanishing gradients.
 ✅ Stronger Signal Retention: Preserves critical information, enabling the model to capture subtle patterns in team performance.



My algorithm assigns win probabilities for each game. As expected, in matchups with a wider skill gap—like a first-round game between Auburn (1 seed) and Alabama State (16 seed)—the higher-seeded team is heavily favored. Auburn, for example, has over a 99% chance of winning in this scenario. While games featuring closer-seeded teams such as those between 8 and 9 seeds have less decisive win probabilities. As the tournament progresses and the skill gap narrows, win probabilities trend closer to 50%, reflecting more competitive matchups. These toss-up games aren’t just harder to predict—they're also the most exciting to watch! 
