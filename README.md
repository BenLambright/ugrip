# ugrip
Baby Cry Update Code
Baby Cry Datasets are on master branch

current plans:
- test temporal contrastive loss function
- Synthetic data using DCASE23 & UrbanSounds8K as background to generate new examples. Tools: Scaper or SpatialScaper.

Ben's code
- transformer architecture that runs very quickly and has an accuracy around 86%, now includes batching
- includes combined IAA files

Danning's code:
- metrics

Mohamed's code:
- CRNN with temporal contrastive loss. Uses Danning and Ben's code as a base to keep the same preprocessing approach to ensure it works with the metrics implemented. Has dimensional mismatch problems due to using MFCC as an input to CNN, almost resolved.
- Custom dataset loader. Great for standardizing data input into our models. Needs to be adapted to use json files (currently uses csv because it's easier to index and retrieve)

audio_trimmed_names data uploaded:
- The names are trimmed to the first 47 characters to avoid conflict.
- This is because orginal Textgrid names converted _ to - at the end of the file, so we trimmed to keep relevant identification and avoiding mismatch.
