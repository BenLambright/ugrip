# ugrip
Baby Cry Update Code

current plans:
- test temporal contrastive loss function
- implement evaluation metrics
- Augment data using DCASE23 & UrbanSounds8K (half-done, only few examples were used)

Ben's code
- transformer architecture that runs very quickly and has an accuracy around 86%
- implements Danning's metrics function so we can get a more accurate accuracy metric (generally around 80%)
- trains on Ben & Saad's annotations, as well as 10 DCASE non-cry audio sets

Danning's code:
- metrics

Mohamed's code:
- CRNN with temporal contrastive loss. Has dimensional mismatch problems due to using MFCC as an input to CNN, almost resolved.
- Custom dataset loader. Great for standardizing data input into our models. Needs to be adapted to use json files (currently uses csv because it's easier to index and retrieve)

BabyCry data uploaded:
- The names are trimmed to the first 47 characters to avoid conflict.
- This is because orginal Textgrid names converted _ to - at the end of the file, so we trimmed to keep relevant identification and avoiding mismatch.
