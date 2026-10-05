# Scientific References & Biological Foundation

NeuroPest models the neurobiology, biophysics, and behavior of the fruit fly (*Drosophila melanogaster*). The underlying neural simulation directly interfaces with the whole-brain connectome from **FlyWire (v783)** and implements empirically validated neural mechanisms across motor control, reinforcement learning, sensory processing, and metabolic regulation.

This document provides a comprehensive bibliography of the scientific literature and datasets utilized in NeuroPest, mapping each study to the corresponding computational modules and biological cell types.

---

## Table of Contents

1. [Whole-Brain Connectome & LIF Biophysics](#1-whole-brain-connectome--lif-biophysics)
2. [Descending Motor Control & Action Selection](#2-descending-motor-control--action-selection)
3. [Mushroom Body Associative Learning & Plasticity](#3-mushroom-body-associative-learning--plasticity)
4. [Metabolism, Hunger & Foraging Drive](#4-metabolism-hunger--foraging-drive)
5. [Visual Perception & Retinotopic Feature Detectors](#5-visual-perception--retinotopic-feature-detectors)
6. [Datasets, Licenses & Attribution](#6-datasets-licenses--attribution)
7. [BibTeX Bibliography](#7-bibtex-bibliography)

---

## 1. Whole-Brain Connectome & LIF Biophysics

NeuroPest's neural substrate is built from the complete electron-microscopy wiring diagram of an adult female *Drosophila melanogaster* brain. Synaptic dynamics follow leaky integrate-and-fire (LIF) equations with biophysically calibrated membrane and synaptic time constants.

- **Dorkenwald et al. (2024)**  
  *Neuronal wiring diagram of an adult brain.*  
  **Nature**, 634(8032), 124–138. [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)  
  *Implementation:* The baseline FlyWire v783 connectome containing 138,639 reconstructed neurons and ~15.1 million synaptic connections (`data/raw/Connectivity_783.parquet`, `neuropest/flywire.py`).

- **Schlegel et al. (2024)**  
  *Whole-brain annotation and multi-connectome cell typing of Drosophila.*  
  **Nature**, 634(8032), 139–152. [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)  
  *Implementation:* Comprehensive cell type classifications, hemispheric flow annotations, neurotransmitter predictions, and descending neuron identifications (`Supplemental_file1_neuron_annotations.tsv`).

- **Shiu et al. (2024)**  
  *A leaky integrate-and-fire computational model based on the adult Drosophila connectome.*  
  **Nature**, 634(8032), 210–219. [doi:10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9)  
  *Implementation:* Biophysical LIF parameters (membrane potential $V_{\text{rest}} = -52\text{ mV}$, spike threshold $V_{\text{th}} = -45\text{ mV}$, reset $V_{\text{reset}} = -55\text{ mV}$, membrane time constant $\tau_m = 20\text{ ms}$, synaptic decay $\tau_s = 5\text{ ms}$, synaptic delay 1.8 ms, and unitary synaptic weight $w = \text{sign} \times 0.275\text{ mV}$). Used in `neuropest/engine/lif.py` and `neuropest/engine/lif_wgpu.py`.

- **Li et al. (2020)**  
  *The connectome of the adult Drosophila mushroom body: implications for function.*  
  **eLife**, 9, e62576. [doi:10.7554/eLife.62576](https://doi.org/10.7554/eLife.62576)  
  *Implementation:* Morphological and compartmental classification of Kenyon cells, Mushroom Body Output Neurons (MBONs), and Dopaminergic Neurons (DANs).

---

## 2. Descending Motor Control & Action Selection

Locomotion and behavioral state selection are decoded directly from identified descending neurons (DNs) that connect the central brain to the ventral nerve cord (VNC).

### Escape & Takeoff (Giant Fiber / DNp01)
- **Card, G., & Dickinson, M. H. (2008)**  
  *Visually mediated motor planning in the escape response of Drosophila.*  
  **Current Biology**, 18(17), 1300–1307. [doi:10.1016/j.cub.2008.07.094](https://doi.org/10.1016/j.cub.2008.07.094)  
  *Implementation:* Kinematics of escape takeoff and leg extension timing.
- **von Reyn, C. R., et al. (2014)**  
  *A spike-timing mechanism for action selection.*  
  **Nature Neuroscience**, 17(7), 962–970. [doi:10.1038/nn.3741](https://doi.org/10.1038/nn.3741)  
  *Implementation:* Two-spike burst threshold for immediate takeoff action triggering (`BrainSpec.gf_event_spikes = 2`).
- **von Reyn, C. R., et al. (2017)**  
  *Feature extraction by a cell-type-specific neural circuit for visual escape in Drosophila.*  
  **Neuron**, 93(4), 844–858. [doi:10.1016/j.neuron.2017.01.034](https://doi.org/10.1016/j.neuron.2017.01.034)  
  *Implementation:* Looming visual pathways linking lobula projection neurons LPLC2 (angular size) and LC4 (expansion velocity) to Giant Fiber dendrites.
- **Ache, J. M., et al. (2019)**  
  *State-dependent visual processing in Drosophila during takeoff.*  
  **Current Biology**, 29(16), 2715–2729. [doi:10.1016/j.cub.2019.07.026](https://doi.org/10.1016/j.cub.2019.07.026)  
  *Implementation:* Visual processing modulation during walking versus flight transitions.
- **Dombrovski, M., Kim, Y. J., & Card, G. M. (2023)**  
  *Neural circuit mechanisms for directional escape steering in Drosophila.*  
  *Implementation:* Lateral LC4 receptive field gradients driving asymmetric activation of DNp02 and DNp11 to direct takeoff away from visual threats (`tools/probe_escape.py`, `neuropest/fly.py`).

### Moonwalker / Backward Walking (MDN)
- **Bidaye, S. S., Machacek, C., Wu, Y., & Dickson, B. J. (2014)**  
  *'Moonwalker' descending neurons that control backward walking in Drosophila.*  
  **Science**, 344(6179), 97–101. [doi:10.1126/science.1249964](https://doi.org/10.1126/science.1249964)  
  *Implementation:* MDN command neuron firing triggering sustained backward walking (`RETREAT` state in `neuropest/brain.py` and `neuropest/fly.py`).
- **Sen, R., et al. (2017)**  
  *Moonwalker descending neurons mediate visually evoked retreat in Drosophila.*  
  **Current Biology**, 27(5), 766–773. [doi:10.1016/j.cub.2017.02.008](https://doi.org/10.1016/j.cub.2017.02.008)  
  *Implementation:* Visual projection neuron LC16 / LPC1 inputs converging onto MDN to trigger defensive retreat under slow visual approach (`tools/probe_retreat.py`).

### Forward Walking Drive (DNp09 / P9)
- **Bidaye, S. S., et al. (2020)**  
  *Two brain pathways initiate and modulate walking in Drosophila.*  
  **Neuron**, 108(4), 694–707. [doi:10.1016/j.neuron.2020.08.016](https://doi.org/10.1016/j.neuron.2020.08.016)  
  *Implementation:* Tonic drive injected into DNp09 (P9) to initiate and sustain forward locomotion; modulated by motility slider, hunger state, and learned reward valence.

### Steering Control (DNa01 / DNa02)
- **Rayshubskiy, A., et al. (2020 / 2025)**  
  *Neural circuit mechanisms for steering control in walking Drosophila.*  
  **eLife**, 14:RP103565. [doi:10.7554/eLife.103565.1](https://doi.org/10.7554/eLife.103565.1)  
  *Implementation:* Ipsilateral DNa02 descending neuron activation generating rotational torque and lateral steering turns toward or away from stimuli (`neuropest/fly.py`, `neuropest/brain.py`).

### Grooming Hierarchy (aDN1 / aDN2)
- **Seeds, A. M., et al. (2014)**  
  *A suppression hierarchy that programs grooming behavior in Drosophila.*  
  **eLife**, 3:e02951. [doi:10.7554/eLife.02951](https://doi.org/10.7554/eLife.02951)  
  *Implementation:* Cephalic grooming prioritized over walking when mechanosensory stimuli are detected on the head.
- **Hampel, S., Franconville, R., Simpson, J. H., & Seeds, A. M. (2015)**  
  *A neural command circuit for grooming movement sequences.*  
  **eLife**, 4:e08758. [doi:10.7554/eLife.08758](https://doi.org/10.7554/eLife.08758)  
  *Implementation:* Head bristle mechanosensory inputs and Johnston's Organ (JO-C/E) driving anterior grooming descending neurons DNg62 (aDN1) and DNge078 (aDN2) (`tools/probe_touch.py`).

### Freezing & Persistent Arousal
- **Zacarias, R., et al. (2018)**  
  *Speed dependent movement paralysis in Drosophila upon looming stimulus.*  
  **eLife**, 7:e37817. [doi:10.7554/eLife.37817](https://doi.org/10.7554/eLife.37817)  
  *Implementation:* Non-escape intermediate looming expansion rate triggering movement arrest / freezing (`FREEZE` state in `neuropest/brain.py`).
- **Gibson, W. T., et al. (2015)**  
  *Behavioral responses to a repetitive visual threat stimulus express a persistent state of defensive arousal in Drosophila.*  
  **Current Biology**, 25(11), 1401–1415. [doi:10.1016/j.cub.2015.03.058](https://doi.org/10.1016/j.cub.2015.03.058)  
  *Implementation:* Defensive arousal slow decay with 30-second exponential time constant (`tau_arousal_s = 30.0` in `neuropest/brain.py`).

---

## 3. Mushroom Body Associative Learning & Plasticity

The Mushroom Body (MB) mediates associative olfactory and visual reinforcement learning via three-factor synaptic plasticity at Kenyon Cell $\to$ MBON synapses.

- **Caron, S. J., Ruta, V., Abbott, L. F., & Axel, R. (2013)**  
  *Random convergence of olfactory inputs in the Drosophila mushroom body.*  
  **Nature**, 497(7447), 113–117. [doi:10.1038/nature12063](https://doi.org/10.1038/nature12063)  
  *Implementation:* Each Kenyon cell dendritic claw samples a small pseudo-random subset (~6 inputs) of projection neurons (`neuropest/mushroom.py`).

- **Turner, G. C., Bazhenov, M., & Laurent, G. (2008)**  
  *Olfactory representations by Drosophila mushroom body neurons.*  
  **Nature Neuroscience**, 11(11), 1259–1267. [doi:10.1038/nn.2214](https://doi.org/10.1038/nn.2214)  
  *Implementation:* Anterior Paired Lateral (APL) feedback inhibition maintaining sparse population coding (~5% active Kenyon cells).

- **Aso, Y., et al. (2014a)**  
  *The neuronal architecture of the mushroom body provides a logic for associative learning.*  
  **eLife**, 3:e04577. [doi:10.7554/eLife.04577](https://doi.org/10.7554/eLife.04577)  
  *Implementation:* Anatomical compartmentalization of the mushroom body lobes into discrete functional units innervated by distinct MBON and DAN types.

- **Aso, Y., et al. (2014b)**  
  *Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila.*  
  **eLife**, 3:e04580. [doi:10.7554/eLife.04580](https://doi.org/10.7554/eLife.04580)  
  *Implementation:* Push-pull valence logic: approach MBONs (e.g. MBON12) vs. avoidance MBONs (e.g. MBON01–04) determining net behavioral valence ($V \in [-1, +1]$).

- **Hige, T., Aso, Y., Modi, M. N., Rubin, G. M., & Turner, G. C. (2015)**  
  *Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila.*  
  **Neuron**, 88(5), 985–998. [doi:10.1016/j.neuron.2015.11.003](https://doi.org/10.1016/j.neuron.2015.11.003)  
  *Implementation:* Three-factor plasticity: coincidence of active Kenyon Cell + Dopamine release depresses corresponding KC $\to$ MBON synaptic weights, skewing the push-pull output.

- **Claridge-Chang, A., et al. (2009)**  
  *Writing memories with light-addressable reinforcement circuitry.*  
  **Cell**, 139(2), 405–415. [doi:10.1016/j.cell.2009.08.034](https://doi.org/10.1016/j.cell.2009.08.034)  
  *Implementation:* PPL1 dopaminergic cluster mediating negative reinforcement (punishment upon screen edge collisions), and PAM cluster mediating sugar/food reward.

- **Berry, J. A., Phan, A., & Davis, R. L. (2018)**  
  *Dopamine neurons promote a clean slate for learning by remembering and forgetting.*  
  **Current Biology**, 28(21), R1245–R1255. [doi:10.1016/j.cub.2018.09.053](https://doi.org/10.1016/j.cub.2018.09.053)  
  *Implementation:* Reward prediction error (RPE) and slow spontaneous synaptic recovery (extinction / forgetting dynamics in `neuropest/mushroom.py`).

---

## 4. Metabolism, Hunger & Foraging Drive

Energy depletion and satiety modulate sensory salience, dopamine release, and motor drive:

- **Krashes, M. J., et al. (2009)**  
  *A neural circuit mechanism integrating hunger and satiety states with odor-reward memory in Drosophila.*  
  **Cell**, 139(2), 416–427. [doi:10.1016/j.cell.2009.08.035](https://doi.org/10.1016/j.cell.2009.08.035)  
  *Implementation:* Hunger state strictly gates PAM dopamine reward neurons and enhances sensitivity to food-related cues. Satiated feeding produces no associative reinforcement, while starved states trigger DNp09-driven foraging hyperactivity (`neuropest/metabolism.py`, `neuropest/brain.py`).

---

## 5. Visual Perception & Retinotopic Feature Detectors

In addition to synthetic mouse cursor coordinates, NeuroPest supports true optical screen capture perception through compound eye geometry:

- **Fischbach, K. F., & Dittrich, A. P. (1989)**  
  *The optic lobe of Drosophila melanogaster. I. A Golgi analysis of wild-type structure.*  
  **Cell and Tissue Research**, 258(3), 441–475. [doi:10.1007/BF00218858](https://doi.org/10.1007/BF00218858)  
  *Implementation:* Medulla column geometry, lamina cell projections (L1–L3), and hexagonal ommatidial mapping (`neuropest/eyebuild.py`).

- **Wu, M., et al. (2016)**  
  *Visual projection neurons in the Drosophila lobula link feature detection to distinct behavioral programs.*  
  **eLife**, 5:e21022. [doi:10.7554/eLife.21022](https://doi.org/10.7554/eLife.21022)  
  *Implementation:* Lobula Columnar (LC) and Lobula Plate/Lobula Columnar (LPLC) projection neurons: LPLC2/LC4 (looming expansion), LC10 (small moving targets/predators), LC16 (retreat) (`neuropest/vision.py`, `neuropest/visual.py`).

- **Isaacson, M. D., Nern, A., Rogers, E. M., & Card, G. M. (2023)**  
  *Visual motion pathways mediating flight deceleration and steering in Drosophila.*  
  **Current Biology**, 33.  
  *Implementation:* LPC1 regressive flow and visual deceleration pathways (`tools/probe_retreat.py`).

---

## 6. Datasets, Licenses & Attribution

| Resource | Origin | License / Terms | Reference |
|---|---|---|---|
| **FlyWire Connectome v783** | Princeton University / MRC LMB | [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) (Non-Commercial Research & Education) | Dorkenwald et al. (2024) |
| **Whole-Brain Annotations** | Cambridge / MRC LMB / FlyConnectome | [CC BY-NC 4.0](https://flywire.ai/guidelines) | Schlegel et al. (2024) |
| **LIF Model & Conductances** | UC Berkeley (Scott Lab) | MIT License | Shiu et al. (2024) |
| **Hemibrain MB Connectome** | Janelia Research Campus | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Li et al. (2020) |

NeuroPest source code is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE) (SPDX: `PolyForm-Noncommercial-1.0.0`) and distributed strictly for non-commercial educational and research purposes in compliance with FlyWire data usage guidelines.

---

## 7. BibTeX Bibliography

```bibtex
@article{dorkenwald2024wiring,
  title={Neuronal wiring diagram of an adult brain},
  author={Dorkenwald, Sven and Matsliah, Arie and Sterling, Philipp and Schlegel, Philipp and Yu, Szi-chieh and McKellar, Claire E and Lin, Amy and Costa, Marta and Eichler, Katharina and Yin, Yijie and Murthy, Mala and Seung, H Sebastian},
  journal={Nature},
  volume={634},
  number={8032},
  pages={124--138},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-024-07558-y}
}

@article{schlegel2024whole,
  title={Whole-brain annotation and multi-connectome cell typing of Drosophila},
  author={Schlegel, Philipp and Yin, Yijie and Bates, Alexander Shakeel and Dorkenwald, Sven and Eichler, Katharina and Brooks, Paul and Han, Doug S and Gkantia, Marina and dos Santos, Markus and Jefferis, Gregory SXE},
  journal={Nature},
  volume={634},
  number={8032},
  pages={139--152},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-024-07686-5}
}

@article{shiu2024leaky,
  title={A leaky integrate-and-fire computational model based on the adult Drosophila connectome},
  author={Shiu, Philip K and Sterne, Gary R and Spiller, Noelle and Romain, Celine P and Nojima, Tetsuya and Cirillo, Matthew and Scott, Kristin},
  journal={Nature},
  volume={634},
  number={8032},
  pages={210--219},
  year={2024},
  publisher={Nature Publishing Group},
  doi={10.1038/s41586-024-07763-9}
}

@article{bidaye2014moonwalker,
  title={'Moonwalker' descending neurons that control backward walking in Drosophila},
  author={Bidaye, Salil S and Machacek, Christian and Wu, Yang and Dickson, Barry J},
  journal={Science},
  volume={344},
  number={6179},
  pages={97--101},
  year={2014},
  publisher={American Association for the Advancement of Science},
  doi={10.1126/science.1249964}
}

@article{bidaye2020two,
  title={Two brain pathways initiate and modulate walking in Drosophila},
  author={Bidaye, Salil S and Laturney, Megan and Chang, Andrew K and Liu, Yichun and Hargreaves, Adam D and Dickson, Barry J},
  journal={Neuron},
  volume={108},
  number={4},
  pages={694--707},
  year={2020},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2020.08.016}
}

@article{card2008visually,
  title={Visually mediated motor planning in the escape response of Drosophila},
  author={Card, Gwyneth and Dickinson, Michael H},
  journal={Current Biology},
  volume={18},
  number={17},
  pages={1300--1307},
  year={2008},
  publisher={Elsevier},
  doi={10.1016/j.cub.2008.07.094}
}

@article{vonreyn2017feature,
  title={Feature extraction by a cell-type-specific neural circuit for visual escape in Drosophila},
  author={von Reyn, Catherine R and Nern, Aljoscha and Williamson, W Ryan and Breads, Patrick and Wu, Ming and Namiki, Shigehiro and Card, Gwyneth M},
  journal={Neuron},
  volume={93},
  number={4},
  pages={844--858},
  year={2017},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2017.01.034}
}

@article{ache2019state,
  title={State-dependent visual processing in Drosophila during takeoff},
  author={Ache, Jan M and Namiki, Shigehiro and Lee, Arthur and Branson, Kristin and Card, Gwyneth M},
  journal={Current Biology},
  volume={29},
  number={16},
  pages={2715--2729},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.cub.2019.07.026}
}

@article{rayshubskiy2025neural,
  title={Neural circuit mechanisms for steering control in walking Drosophila},
  author={Rayshubskiy, Aleksandr and Holtz, Stephen L and D'Alessandro, Isabella and Li, An-An and Vanderbeck, Quentin X and Haber, Isabel S and Wilson, Rachel I},
  journal={eLife},
  volume={14},
  pages={RP103565},
  year={2025},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.103565.1}
}

@article{hampel2015neural,
  title={A neural command circuit for grooming movement sequences},
  author={Hampel, Stefanie and Franconville, Romain and Simpson, Julie H and Seeds, Andrew M},
  journal={eLife},
  volume={4},
  pages={e08758},
  year={2015},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.08758}
}

@article{aso2014neuronal,
  title={The neuronal architecture of the mushroom body provides a logic for associative learning},
  author={Aso, Yoshinori and Hattori, Daisuke and Yu, Yi and Johnston, Rebecca M and Iyer, Nirmala A and Ngo, Teri-TB and Rubin, Gerald M},
  journal={eLife},
  volume={3},
  pages={e04577},
  year={2014},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.04577}
}

@article{aso2014mbon,
  title={Mushroom body output neurons encode valence and guide memory-based action selection in Drosophila},
  author={Aso, Yoshinori and Sitaraman, Divya and Ichinose, Toshiharu and Kaun, Karla R and Vogt, Katrin and Tanimoto, Hiromu and Rubin, Gerald M},
  journal={eLife},
  volume={3},
  pages={e04580},
  year={2014},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.04580}
}

@article{hige2015heterosynaptic,
  title={Heterosynaptic plasticity underlies aversive olfactory learning in Drosophila},
  author={Hige, Tatsuya and Aso, Yoshinori and Modi, Mehrab N and Rubin, Gerald M and Turner, Glenn C},
  journal={Neuron},
  volume={88},
  number={5},
  pages={985--998},
  year={2015},
  publisher={Elsevier},
  doi={10.1016/j.neuron.2015.11.003}
}

@article{krashes2009neural,
  title={A neural circuit mechanism integrating hunger and satiety states with odor-reward memory in Drosophila},
  author={Krashes, Michael J and DasGupta, Suewei and Sengupta, Ananya and Grosjean, Yael and Florman, Zachary M and Waddell, Scott},
  journal={Cell},
  volume={139},
  number={2},
  pages={416--427},
  year={2009},
  publisher={Elsevier},
  doi={10.1016/j.cell.2009.08.035}
}

@article{caron2013random,
  title={Random convergence of olfactory inputs in the Drosophila mushroom body},
  author={Caron, Sophie JC and Ruta, Vanessa and Abbott, LF and Axel, Richard},
  journal={Nature},
  volume={497},
  number={7447},
  pages={113--117},
  year={2013},
  publisher={Nature Publishing Group},
  doi={10.1038/nature12063}
}

@article{turner2008olfactory,
  title={Olfactory representations by Drosophila mushroom body neurons},
  author={Turner, Glenn C and Bazhenov, Maxim and Laurent, Gilles},
  journal={Nature Neuroscience},
  volume={11},
  number={11},
  pages={1259--1267},
  year={2008},
  publisher={Nature Publishing Group},
  doi={10.1038/nn.2214}
}

@article{claridge2009writing,
  title={Writing memories with light-addressable reinforcement circuitry},
  author={Claridge-Chang, Adam and Roorda, Robert D and Vrontou, Eftychia and Sjulson, Lucas and Li, He and Hirsh, Jay and Anderson, David J},
  journal={Cell},
  volume={139},
  number={2},
  pages={405--415},
  year={2009},
  publisher={Elsevier},
  doi={10.1016/j.cell.2009.08.034}
}

@article{wu2016visual,
  title={Visual projection neurons in the Drosophila lobula link feature detection to distinct behavioral programs},
  author={Wu, Ming and Nern, Aljoscha and Williamson, W Ryan and Morimoto, Mai M and Reiser, Michael B and Card, Gwyneth M and Rubin, Gerald M},
  journal={eLife},
  volume={5},
  pages={e21022},
  year={2016},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.21022}
}

@article{zacarias2018speed,
  title={Speed dependent movement paralysis in Drosophila upon looming stimulus},
  author={Zacarias, Raquel and Namiki, Shigehiro and Card, Gwyneth M and Vasconcelos, Maria Luisa and Moita, Marta A},
  journal={eLife},
  volume={7},
  pages={e37817},
  year={2018},
  publisher={eLife Sciences Publications Limited},
  doi={10.7554/eLife.37817}
}
```
