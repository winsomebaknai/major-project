from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('./checkpoints/Test_5KModel', 'checkpoint_best.pt', tokenizer='moses', bpe='fastbpe', bpe_codes="./checkpoints/Test_5KModel/codes")
custom_lm.cuda()
result = custom_lm.sample('दा मोनायाव', beam=5)
file1 = open('./external_files/output.txt','w')
file1.write(result)
print(result)
