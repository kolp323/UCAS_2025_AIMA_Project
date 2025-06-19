# 训练好的模型保存与调用说明

## 1. 训练好的模型保存在哪里？

训练好的模型会保存在 `runs/` 目录下的子文件夹中，具体路径和文件名格式如下：

- **预训练模型（pretrain）**  
  ```
  runs/seq2seq/exp/checkpoint_pretrain_epoch_{epoch}.pth
  ```
  例如：`runs/seq2seq/exp/checkpoint_pretrain_epoch_3.pth`

- **强化学习训练模型（train）**  
  ```
  runs/seq2seq/exp/checkpoint_train_mod4_step_{step}.pth
  ```
  例如：`runs/seq2seq/exp/checkpoint_train_mod4_step_2000.pth`

模型保存代码示例：
```python
trainer.save_checkpoint(tag='pretrain_epoch_'+str(epoch))
```

---

## 2. 怎么调用训练好的模型？

加载模型的典型代码如下：

```python
from pensmodule.Generator import *
model_path = '../../runs/seq2seq/exp/checkpoint_train_mod4_step_2000.pth'

def load_model_from_ckpt(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    if torch.cuda.device_count() > 1:
        print('multiple gpu training')
        model = nn.DataParallel(model)
    return model

model = load_model_from_ckpt(model_path).to(device)
model.eval()
```

---

## 3. 输出的预测结果在哪里？

预测输出通过 `predict` 函数生成并返回，示例代码如下：

```python
from pensmodule.Generator.eval import predict
refs, hyps, scores1, scores2, scoresf = predict(
    usermodel, model, test_iter, device, index2word, beam=False, beam_size=3, eos_id=2
)
```

- `refs`：真实标题
- `hyps`：模型生成的标题（预测结果）
- `scores1, scores2, scoresf`：ROUGE-1、ROUGE-2、ROUGE-L 分数

如需保存预测结果到文件，可添加如下代码：

```python
with open('predictions.txt', 'w', encoding='utf-8') as f:
    for ref, hyp in zip(refs, hyps):
        f.write(f"REF: {ref}\nHYP: {hyp}\n\n")
```

---

## 相关文件和函数

- `runs/seq2seq/exp/`（模型保存目录）
- `Trainer.save_checkpoint`
- `load_model_from_ckpt`
- `predict`
- `pipeline_pretrain_train_test.ipynb`（完整流程示例）

如需进一步保存或处理预测结果，可在 notebook 中自行添加保存逻辑。