from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('./checkpoints/Test_5KModel', 'checkpoint_best.pt', tokenizer='moses', bpe='fastbpe', bpe_codes="./checkpoints/Test_5KModel/codes")
custom_lm.cuda()
result = custom_lm.sample('बे जेब्लाबो', beam=5)
print(result)
