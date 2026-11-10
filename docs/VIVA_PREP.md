# AgriSense: viva / Q&A preparation

Short answers to the questions examiners are most likely to ask. Irrigation numbers are from
`results/summary_by_policy.csv` (60 unseen test seasons, 2021-2024); leaf-disease numbers are from
`results/leaf_metrics.json` (1,213 held-out test images).

Live site: <https://nirmit10000.github.io/agrisense/>

## The 30-second pitch

> AgriSense answers a farmer's two daily questions: *when should I water?* and *what's wrong with my
> crop?* For irrigation, we trained a Q-learning agent in a field simulator built on the FAO-56
> standard, driven by 16 years of real weather from Pune, Jodhpur and Ludhiana. On 60 seasons it never
> saw, it used **45% less water than a fixed schedule with higher yield (98.1% vs 93.7%)**, and slightly
> less water than an expert soil-sensor rule. For crop health, a MobileNetV2 CNN trained by transfer
> learning on PlantVillage recognises 14 maize and tomato leaf classes with **93.7% test accuracy**, and
> a heatmap shows where on the leaf it looked. Both run in one website, with the disease model running on
> the user's own device.

## Numbers to remember

| | RL agent | Sensor rule | Fixed schedule | Rainfed |
|---|---|---|---|---|
| Water / season | **435 mm** | 460 mm | 784 mm | 0 |
| Yield (% of max) | 98.1 % | 100 % | 93.7 % | 31.8 % |
| Water wasted to drainage | 1.6 mm | 4.2 mm | 354 mm | 0 |
| kg of crop per m³ water | **6.30** (best) | 6.13 | 3.33 | - |

360 states · 5 actions · 50,000 training seasons per crop · ~1.5 min training per crop · 3 crops · 3 cities.

---

## Problem and motivation

**Q: Why is this an AI problem and not just a formula?**
Because the decision is *sequential and uncertain*. Watering today changes tomorrow's soil moisture,
rain may or may not come, and the payoff (yield) is only known at season end. The FAO formulas tell us
how water moves; they don't tell us the best *schedule* under an uncertain forecast. That's what the
agent learns.

**Q: Why reinforcement learning rather than supervised learning?**
Supervised learning needs labelled examples of the "correct" irrigation, and no such dataset exists.
RL learns from consequences (reward), which we can compute in a simulator.

## Data

**Q: Where does the data come from? Is it real?**
Real daily weather (rain, reference evapotranspiration ET0, temperatures) for 2010-2025 from the
Open-Meteo Historical Weather API, which serves the ERA5 reanalysis from ECMWF. The *field* is simulated,
but the *weather driving it* is real.

**Q: Why these three cities?**
To maximise climatic variety so the agent can't memorise one place: Pune has a heavy monsoon
(~1120 mm/yr), Jodhpur is desert (~430 mm/yr, highest ET0), and Ludhiana has cold winters with some
winter rain (~780 mm/yr).

**Q: How did you split train and test?**
By time: seasons starting 2010-2020 for training, 2021-2024 for testing. A random split would leak
information, because neighbouring days are correlated.

## The simulator

**Q: What is FAO-56 and why trust it?**
FAO Irrigation and Drainage Paper 56 (Allen et al., 1998) is the international standard method for crop
water requirements, used by irrigation engineers worldwide. We use its single-crop-coefficient daily
soil-water balance.

**Q: Explain TAW, RAW, Kc, Ks in one line each.**
- TAW: how much water the root zone can hold between full (field capacity) and empty (wilting point).
- RAW = p × TAW: the part the crop can use without stress.
- Kc: crop coefficient, which scales the weather's drying power (ET0) to this crop at this stage.
- Ks: stress factor, which is 1 while depletion ≤ RAW and falls linearly to 0 as the soil empties.

**Q: How is yield calculated?**
FAO-33: `1 − Ya/Ym = Ky (1 − ΣETa/ΣETc)`. If the crop got 90% of the water it wanted and Ky = 1.25,
it loses 12.5% yield.

## RL design

**Q: What exactly is the state? Why those four features?**
Soil dryness (10 levels), growth stage (4), 3-day rain forecast (none/light/heavy), and today's ET0
(low/medium/high), giving 10×4×3×3 = 360 states. These are exactly what a farmer considers: how dry is it,
how thirsty is the crop now, is rain coming, how hot is it.

**Q: Why discrete actions 0-40 mm?**
They are realistic drip or sprinkler doses, and on the same scale as the crop's readily available water
(RAW is about 29-53 mm depending on the crop), so one large dose roughly refills the safe zone. Finer steps
would multiply the table size for little gain.

**Q: Explain the reward.**
Per day: −(1 × water used + 10 × crop water shortage + 1 × water drained away + 10 per irrigation event).
Water costs money; shortage costs yield, which matters ~10× more; drainage is pure waste; the event cost
represents pump energy and labour and stops the agent making many tiny irrigations.

**Q: How did you choose the weights? Aren't they arbitrary?**
They encode priorities, which is a design choice in any RL problem. We tried a stress weight of 5 and of
10. With 5, the agent saved more water but yield dropped to ~97%; 10 gave a better balance (98%+).
A water-scarce region could raise the water cost. It's one line in `config.py`.

**Q: Why tabular Q-learning and not DQN / PPO?**
The state space is small (360), so a table learns it exactly, trains in ~1.5 minutes on a CPU, and,
most importantly, can be printed and inspected (slide 12). A neural network would be a black box with no
accuracy benefit at this scale. DQN is our listed next step for continuous inputs.

**Q: Write the Q-learning update and explain each term.**
`Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') − Q(s,a)]`
α = learning rate (how far to move towards the new estimate), r = today's reward, γ = discount (how much
tomorrow matters; 0.95), max Q(s′,·) = value of acting best from tomorrow's state. The bracket is the
TD error: how surprised we were.

**Q: Why decay α and ε?**
ε (exploration) decays 1 → 0.05 so the agent explores early and exploits later. α decays 0.1 → 0.01 so
values settle. With a constant α, actions with almost equal value kept swapping, and the policy table
looked noisy.

**Q: Is Q-learning on-policy or off-policy?**
Off-policy: it learns the value of the *greedy* policy while behaving ε-greedily.

**Q: How do you know it converged?**
The training curve (slide 7) rises and then is flat for the last ~15,000 episodes, and test results are
stable.

## Results

**Q: The sensor rule gets 100% yield and yours 98%. Isn't the rule better?**
It's a trade-off, not a loss. The rule refills the soil to 100% every time, which is safe for yield but
uses more water and wastes more when rain follows. Our agent uses 5.5% less water and has the best water
productivity (kg per m³). Also, the rule needs a perfect moisture sensor and hand-coded agronomy; the
agent *discovered* a near-identical threshold (~60% vs FAO's 55%) on its own.

**Q: Where does the water saving come from?**
Three things: (1) it uses the rain forecast and skips or reduces irrigation before rain; (2) it applies
smaller 30-40 mm doses, leaving room in the soil for rain instead of filling to the brim; (3) compared
with the fixed schedule, it simply doesn't irrigate when the soil is still moist.

**Q: Why is Ludhiana weaker (95.4% yield)?**
Our likely explanation (not separately tested): its cold winters produce long runs of low-ET0 days, and
the coarse 10-level dryness bins and 3-level ET0 bins lose detail there, so the agent sometimes waits too
long. That is the main limitation of the tabular approach, and the motivation for a DQN with continuous
inputs.

**Q: What does "rare" mean in the policy table?**
Situations seen fewer than 50 times in training (e.g. soil 90% dry with heavy rain forecast). We don't
interpret those cells, because the agent has too little experience there.

**Q: Is the comparison fair?**
Yes: every policy runs on the same 60 test seasons with identical weather *and identical forecast
errors* (same random seed per season). The sensor rule even gets advantages the agent doesn't have:
perfect moisture knowledge and any irrigation amount, not just 0/10/20/30/40.

## Limitations (say these before they ask)

- Simulated field, not a physical trial.
- FAO-33 yield model is seasonal: stress at flowering isn't weighted more.
- Synthetic forecast (true rain ± 25% error).
- One soil type.
- Coarse discretisation (Ludhiana).
- Leaf model: trained on lab photos (plain background); field photos are harder; only 14 classes, no
  "unknown" option.

**Q: How would the irrigation agent work on a real farm?**
Replace the simulator's soil moisture with a soil-moisture sensor reading, take the forecast from a
weather API, look up the action in the Q-table, and send it to a drip controller. The policy is a
lookup table, so it could even run on a microcontroller.

---

## Leaf disease module

**Leaf numbers to remember** (from `results/leaf_metrics.json`): 14 classes (maize + tomato) ·
train / val / test images **5,660 / 1,213 / 1,213** · test accuracy **93.7%** · macro F1 **0.937** ·
input **224** px · head epochs **40** + fine-tune epochs **3** · training ~**11.5** min on CPU ·
browser model **8.9 MB** · 28/28 demo samples correct in the browser (~65 ms each).

**Q: Why a CNN for this?**
Disease symptoms are visual patterns (spots, lesions, discoloured patches, mould). A convolutional
neural network learns such patterns directly from pixels: early layers detect edges and colours,
later layers detect textures and lesion shapes. Hand-crafted features (colour histograms etc.) would
need an expert to design them and usually do worse.

**Q: Why transfer learning instead of training from scratch?**
We have thousands of images, not millions. A CNN trained from scratch on that would over-fit. Starting
from MobileNetV2 already trained on ImageNet (1.2 million photos) reuses general visual features, so we
only teach the later layers what *leaf diseases* look like. It gets high accuracy with little data and
trains in minutes on a laptop CPU.

**Q: What exactly did you train?**
Two stages. (1) Freeze the pre-trained convolutional layers and train only a new linear classifier
for our 14 classes. (2) Unfreeze the last convolutional blocks and fine-tune them together with the
classifier at a small learning rate, so the high-level features adapt to leaves. Fine-tuning everything
from the start with a large learning rate would destroy the useful pre-trained features.

**Q: Why MobileNetV2 and not ResNet / VGG / a bigger model?**
It is designed for phones (depthwise-separable convolutions, about 3.5M parameters), so it runs inside a
web browser on an ordinary laptop or phone, which is where a farmer would use it. On a clean dataset
like PlantVillage a bigger model gains very little accuracy but is 5-40× larger. Its last layers are
global average pooling plus one linear layer, which makes class activation maps exact (next question).

**Q: What is the heatmap (CAM) and why should we trust it?**
Class activation map (Zhou et al., 2016). The last conv layer produces K feature maps; the classifier
gives each one a weight per class. The CAM for class c is the weighted sum Σ w(c,k) · f_k(x,y). Because
the class score *is* the average of this map, it is not a guess: it shows exactly how much each region
contributed to the decision. If the bright area is on the lesions, the model uses the right evidence;
if it is on the background, we would know the model is cheating. It's coarse (one cell per 32×32 pixels).

**Q: How did you split the data, and why that way?**
70 / 15 / 15 train / validation / test, *stratified* so every class has the same share in each split.
Validation is for choices during training; the test set is used once, at the end, so the reported
accuracy is unbiased. Images per class were capped so that no class dominates and training stays fast.

**Q: What does 93.7% accuracy actually mean?**
Of the 1,213 test leaves the model never saw during training, it named the correct class for 93.7%.
We also report macro F1 (0.937), which weights every class equally, so a weak rare class can't hide behind
the big ones. The per-class table and confusion matrix show which diseases get confused (typically
diseases with similar-looking spots on the same crop).

**Q: Which classes are weakest, and why?**
Three tomato diseases: Spider mites (F1 0.85, recall 73%), Early blight (F1 0.86, recall 79%) and
Target spot (F1 0.86, precision 77%). Spider mites and Early blight are mostly mistaken for Target spot:
all three show small dark or speckled spots on the leaf. Maize classes are much easier.

**Q: Published PlantVillage models get 99%. Why only 93.7%?**
We trained for 11.5 minutes on a laptop CPU, fine-tuning only the last 4 blocks for 3 epochs. Fine-tuning
the whole network for more epochs on a GPU is the obvious next step and would close most of the gap. We
chose a small model that runs on a phone over a big one.

**Q: Would it work on photos from a real field?**
Not as well, and we say so. PlantVillage photos are single leaves on a plain background in good light.
Field photos have soil, other leaves, shadows and blur. In the original PlantVillage deep-learning study
(Mohanty et al., 2016), accuracy fell to about 31% on images taken under different conditions. The model also only knows 14 classes, so any other image is forced into one of them.

**Q: How would you improve it for field use?**
Fine-tune on field datasets (e.g. PlantDoc), add augmentation with real backgrounds and lighting, add an
"unknown / not a leaf" option using a confidence threshold or out-of-distribution detection, crop the
leaf first with a detector, and collect local images from Indian farms.

**Q: Why does the model run in the browser instead of on a server?**
Privacy (the photo never leaves the phone), cost (no server to pay for; free GitHub Pages hosting) and
offline-friendliness (once the page and model are loaded, inference needs no connection). The trained
PyTorch model is exported to ONNX and run by ONNX Runtime Web (WebAssembly).

**Q: Why maize and tomato, and not wheat?**
They match the irrigation crops, so both modules serve the same farmer. PlantVillage has no wheat
images.

---

## Website

**Q: Why a static website and not Streamlit?**
GitHub Pages hosts static files for free and never sleeps, so the link always works during the
presentation. A static site has no server to crash or pay for. The Streamlit app is still in the repo
for local experiments (`streamlit run app.py`).

**Q: How can a CNN run on a static site with no server?**
The model is a file like any image. The page downloads it once (8.9 MB) and ONNX Runtime Web runs the
network in the browser using WebAssembly. Preprocessing (resize, normalise), the softmax and the CAM
heatmap are a few lines of JavaScript.

**Q: Why are the irrigation results precomputed instead of simulated live?**
Every combination (3 cities × crops × sowing seasons × years × 4 policies) was simulated in Python and
saved as JSON. That guarantees the website shows exactly the same numbers as the report, with no risk
of a JavaScript re-implementation drifting from the Python simulator, and every choice responds
instantly. The Q-table itself is also exported, so the rulebook view is the real trained policy.

**Q: Is anything sent to a server?**
No user data. The page loads static files (HTML, JSON, the model) from GitHub Pages and libraries from a
CDN. Leaf photos are processed only on the user's device.
