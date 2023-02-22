from fairseq.models.transformer_lm import TransformerLanguageModel
custom_lm = TransformerLanguageModel.from_pretrained('./checkpoints/Test_5KModel', 'checkpoint_best.pt', tokenizer='moses', bpe='fastbpe', bpe_codes="./checkpoints/Test_5KModel/codes")
custom_lm.cuda()

result = custom_lm.sample('हाबाब आदा', beam=5)
file1 = open('./external_files/output.txt','a')
file1.write(f"\n{result}")
print(result)

result = custom_lm.sample('माबा', beam=5)
file1 = open('./external_files/output.txt','a')
file1.write(f"\n{result}")
print(result)

result = custom_lm.sample('जोबोर', beam=5)
file1 = open('./external_files/output.txt','a')
file1.write(f"\n{result}")
print(result)

result = custom_lm.sample('मानसि', beam=5)
file1 = open('./external_files/output.txt','a')
file1.write(f"\n{result}")
print(result)

result = custom_lm.sample('आच्छा', beam=5)
file1 = open('./external_files/output.txt','a')
file1.write(f"\n{result}")
print(result)

file1.close()
