# %%
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
import mlflow

# %%
device='cuda' if torch.cuda.is_available() else 'cpu'

# %%
response=requests.get("https://raw.githubusercontent.com/karpathy/ng-video-lecture/refs/heads/master/input.txt")

# %%
print(response.text)

# %%
print(response.text[:1000])

# %%
print(len(response.text))

# %%
chars=sorted(list(set(response.text)))
vocab_size=len(chars)
print("".join(chars))
print(vocab_size)

# %%
stoi={ch:i for i,ch in enumerate(chars)}
itos={i:ch for i,ch in enumerate(chars)}
encode=lambda s:[stoi[c] for c in s]
decode=lambda l:''.join([itos[i] for i in l])



print(encode("hii there"))
print(decode(encode("hii there")))

# %%
data=torch.tensor(encode(response.text),dtype=torch.long)
print(data.shape,data.type)
print(data[:1000])

# %%
n=int(.9*len(data))

# %%
train_data=data[:n]
val_data=data[n:]

# %%
block_size=8
train_data[:block_size+1]

# %%
x=train_data[:block_size]
y=train_data[1:block_size+1]
for t in range(block_size):
    context=x[:t+1]
    target=y[t]

    print(f"when input is {context} the target: {target}")

# %%
torch.manual_seed(1337)
batch_size=4
block_size=8

def get_batch(split):
    data=train_data if split=="train" else val_data
    ix=torch.randint(len(data)-block_size,(batch_size,))
    x=torch.stack([data[i:i+block_size] for i in ix])
    y=torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y=x.to(device),y.to(device)
    return x,y
xb,yb=get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


print('-------')


for b in range(batch_size):
    for t in range(block_size):
        context=xb[b,:t+1]
        target=yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")

# %%
torch.manual_seed(1337)

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets=None):
        logits=self.token_embedding_table(idx)
        if targets is None:
            loss=None
        else: 
            B,T,C=logits.shape
            logits=logits.view(B*T,C)
            targets=targets.view(B*T)
            loss=F.cross_entropy(logits,targets)
        return logits,loss
    

    def generate(self,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            logits,loss=self(idx)
            logits=logits[:,-1,:]
            probs=F.softmax(logits,dim=-1)
            idx_next=torch.multinomial(probs,num_samples=1)
            idx=torch.cat((idx,idx_next),dim=1)

        return idx



model=BigramModel(vocab_size=vocab_size).to(device)
logits,loss=model(xb,yb)

print(logits.shape)
print(loss)
idx=torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(model.generate(idx,max_new_tokens=100)[0].tolist()))

# %%
optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)

# %%
batch_size=32

for steps in range(100000000):
    xb,yb=get_batch('train')
    logits,loss=model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if(steps%10000 ==0):
        print(steps)
print(loss.item())

# %%
print(decode(model.generate(idx,max_new_tokens=100)[0].tolist()))

# %%
torch.save(model,"model.pkl")

# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%


# %%



