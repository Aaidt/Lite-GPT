<div align="center">

```
██╗     ██╗████████╗███████╗     ██████╗ ██████╗ ████████╗
██║     ██║╚══██╔══╝██╔════╝    ██╔════╝ ██╔══██╗╚══██╔══╝
██║     ██║   ██║   █████╗      ██║  ███╗██████╔╝   ██║   
██║     ██║   ██║   ██╔══╝      ██║   ██║██╔═══╝    ██║   
███████╗██║   ██║   ███████╗    ╚██████╔╝██║        ██║   
```

> *A clean, educational decoder-only Transformer language model (~25M parameters) trained from scratch on a single NVIDIA A5000 GPU.*

---

</div>

### Architecture

```text
Input Tokens [B, T]
        │
        ▼
┌─────────────────────┐
│  Token Embeddings   │
│  [vocab, d_model]   │
└─────────────────────┘
           │
           │
           ▼
╔══════════════════════════════╗
║ Transformer Block × 8        ║
║                              ║
║       RMSNorm                ║
║           │                  ║
║           ▼                  ║
║     GQA flash Attention      ║
║           │                  ║
║           ▼                  ║
║       Residual Add           ║
║           │                  ║
║           ▼                  ║
║       RMSNorm                ║
║           │                  ║
║           ▼                  ║
║        SwiGLU MLP            ║
║           │                  ║
║           ▼                  ║
║      Residual Add            ║
╚══════════════════════════════╝
            │
            ▼
┌─────────────────────┐
│   Final RMSNorm     │
└─────────────────────┘
            │
            ▼
┌─────────────────────┐
│      LM Head        │
└─────────────────────┘
            │
            ▼
      Logits [B,T,V]
```

| Configuration | Value |
|---|---|
| Model type | Decoder-only Transformer |
| Parameters | ~24.6M |
| Layers | 8 |
| d_model | 448 |
| Attention heads | 8 query / 4 key-value (GQA, head_dim=56) |
| FFN dimension | 1152 (SwiGLU) |
| Context length | 512 |
| Vocabulary | 16,384 (custom BPE tokenizer) |

The model incorporates modern LLM improvements over GPT-2:
- **RoPE** positional encodings
- **Grouped Query Attention (GQA)** with FlashAttention
- **SwiGLU** feed-forward networks
- **RMSNorm** with pre-norm residual connections
- **Weight tying** between token embeddings and LM head

### Datasets

Trained on ~1B tokens (custom BPE tokenizer, vocab 16,384):

| Dataset | Tokens | Weight |
|---|---|---|
| FineWeb | 300M | 60% |
| TinyStories | 200M | 40% |

Data is tokenized with a custom ByteLevel BPE tokenizer (vocab size 16,384), stored as `uint16` arrays, and split 90/10 train/validation.

### Training

| Hyperparameter | Value |
|---|---|
| Effective batch size | 128 |
| Sequence length | 512 |
| Learning rate | 6e-4 → 6e-5 (cosine decay) |
| Warmup steps | 4000 |
| Max steps | 40000 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Precision | BF16/FP16 mixed |
| Tokens seen | ~1B |

Trained on a **NVIDIA A5000 GPU** (Runpod) using Cross Entropy Loss with cosine LR schedule and periodic checkpointing.

### Results

| Metric | Value |
|---|---|
| Train Loss | 2.6853755712509155 |
| Val Loss | 2.8251807928085326 |
| Perplexity | 16.863993599262944 |




# LiteGPT Evaluation Suite
**Generated:** 2026-06-26 19:48:15

---

## Language modeling
- **Settings:** temperature=0.8, top_k=50, max_tokens=100

### Prompt: The capital of France is
- **Output:** The capital of France is well known for its great geographical experience. The historical setting is not one of a sort of the greatest of all the times, but it's a central part of the history of the country. It's a historical town that has seen a huge shift inside this famous city of the same name. The city is known as "The Dolceus" and also known as the "The Dolceus." It's a place where a lot of people live and live,

### Prompt: Once upon a time, there lived
- **Output:** Once upon a time, there lived a little girl named Lily. She was very excited because she was going on a trip to see Grandma. She helped her mom pack their luggage with clothes and toys. Then, they got in the car and drove to Grandma's house.  As they drove, Lily saw something strange. It was a big, gray cloud that seemed to float away. She pointed and asked her mom what it was. Her mom said it was a mild cloud, but it was a very hot day.  

### Prompt: The theory of evolution states that
- **Output:** The theory of evolution states that a single model is responsible for some type of behavior, not all animals, not all human beings. It is not the same concept to decide the same. It does not matter if it is the same in a species or a family, or the exact same animal. When it is no longer possible to choose one species, it is only a matter of determining the kind of a single species. |We can try to make a simple concept based on a model, name or model (or model as

### Prompt: Machine learning is
- **Output:** Machine learning is part of the same, but the reason is that each semester is different, and the number of students attending is different. This is the reason for course selection and inclusion among those at lower-level learning. So, we need to make sure we understand what the learning system is, how it is being presented and what we use to help others find their own kindergarten to finish it. So, we will get to see how each is different. We will see what our teachers and

---

## Reasoning
- **Settings:** temperature=0.4, top_k=40, max_tokens=100

### Prompt: If John has 5 apples and gives 2 to Alice, then John has
- **Output:** If John has 5 apples and gives 2 to Alice, then John has 5 apples and 1 to Bob, then Alice is 3 to Bob, then Alice is 3 to Bob and Bob is 3 to Bob. A few days ago, I made a couple of apples and added them to the bottom of the apple. I think I did a pretty good job with the apple, but I didn't have a lot of apples, so I just added the apples and added the apples. I made a few more apples, but I think I would add a lot more apples

### Prompt: The next number in the sequence is: 2, 4, 8, 16,
- **Output:** The next number in the sequence is: 2, 4, 8, 16, 17, 17, 17, 17, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19, 19,

### Prompt: Earth is to Solar System as electron is to
- **Output:** Earth is to Solar System as electron is to be able to generate a high-level of energy and solar power. The solar system is designed to generate a high-level of energy, which is then used to generate a high-level of energy. Solar systems can generate electricity, which is then used to generate a high-level of energy. Solar systems are used to generate electricity, which is then used to generate electricity. Solar systems are used to generate electricity. The solar system is used to generate electricity which is then

---

## World knowledge
- **Settings:** temperature=0.4, top_k=40, max_tokens=100

### Prompt: Python is a programming language that
- **Output:** Python is a programming language that is used to communicate to the audience. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into English. The language is translated into

### Prompt: The Internet works because
- **Output:** The Internet works because it’s a platform that helps people with disabilities. What is a mobile app? A mobile app is a platform that allows people to use it to work, work, and play. It allows people to use it to work, and it allows them to use it to work. What is a mobile app? A mobile app is a platform that allows people to use it to work, work, and play. It is a platform that allows people to use it to work, work

### Prompt: The largest planet in our solar system is
- **Output:** The largest planet in our solar system is the Earth's solar system. It's a great solar system for solar panels and solar panels. It's also a great solar power for solar panels and solar panels. The solar system is a great solar system for solar panels and solar panels. It's also a great solar system for solar panels and solar panels. The solar system is a great solar power system for solar panels and solar panels. It's also a great solar power system for solar panels and solar panels. The solar system is

---

## Coding
- **Settings:** temperature=0.2, top_k=20, max_tokens=150

### Prompt: def factorial(n):
- **Output:** def factorial(n): a) the use of the term "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "condition" in the context of a "cond

### Prompt: class LinkedList:
- **Output:** class LinkedList: - LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: LinkedList: Linked

### Prompt: for i in range(10):
- **Output:** for i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(10): - i in range(

### Prompt: #include <iostream>  int main() {
- **Output:** #include <iostream>  int main() {\iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS/iOS

### Prompt: function quickSort(arr) {
- **Output:** function quickSort(arr) {iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone(s) iphone

---

## HTML
- **Settings:** temperature=0.2, top_k=20, max_tokens=150

```text
### Prompt: <!DOCTYPE html> <html>
- **Output:** <!DOCTYPE html> <html><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a><a>
```

---

## Markdown
- **Settings:** temperature=0.8, top_k=50, max_tokens=100

### Prompt: # Neural Networks
- **Output:** # Neural Networks. There are several networks that provide the same network as the other networks. At the same time, a network can provide a network with the same network as a wireless network (SMT). What's the difference between VoIP networks? The VoIP network has an IP address that connects both networks (i.e. IP address and IP address) and the Internet address. The network is the network's IP address that connects to a certain network network. This connects

---

## Story completion
- **Settings:** temperature=0.9, top_k=100, max_tokens=200

### Prompt: The dragon opened its eyes and
- **Output:** The dragon opened its eyes and looked around. It saw something shiny and colorful. It jumped and grabbed the object with its sharp claws.  The dragon flew away and the dragon breathed a sigh of joy. It flew back up into the sky and found its owner, who was so proud of it. He said the dragon had been an amazing hunter.

### Prompt: The detective walked into the abandoned warehouse and
- **Output:** The detective walked into the abandoned warehouse and turned right, halfway next to the counter—the warehouse floor was all finished. I wasn’t sure what the guy was doing in front of the counter, but I was having a very slight, deliberate grimacing moment. This might be the first time I saw an account from the FBI for a meeting with a customer, or someone from the FBI’s office. This was my first time seeing the FBI’s staffer work at an incident—one that is usually a business or a place of business. My goal is to solve the problem, and there’s not as much fun as I think. A lot of people that I’m not interested in hearing from have been in the FBI’s management room, and it’s been very, very good. When I needed to tell them where it was going, I was really drawn to a number of good habits, but still had to be careful. There’s been some

---

## Dialogue
- **Settings:** temperature=0.9, top_k=100, max_tokens=200

### Prompt: Alice: Hello! Bob:
- **Output:** Alice: Hello! Bob: I'm sorry. Wow. You're a little bit of an understatement to me. Maybe. I know how to keep you up by yourself, even though I need to share my stories. I know, I found myself having trouble finding relationships, since I'm usually with my best friend or coworkers. I know I've been in relationship situations since I last met my friend from college and I understand I can help. I know I have found the best relationship possible for a couple of weeks now, but it's actually all about finding love. The two of you and your best friend, Mildred P. Stewart (Pennsylvania Girl). That's where the two are: a close relationship, a relationship, and an intercourse. I love it when I'm with her, and her love. These two have just been on a good track. But lately, I actually have found several relationships. "To be truly love means to be happy,

---

## TinyStories style
- **Settings:** temperature=0.9, top_k=100, max_tokens=200

### Prompt: Tom had a little red ball.
- **Output:** Tom had a little red ball. Lily had a red ball and Tom had a blue ball. They both loved to play with their toys. But they both wanted the red ball. They both pulled it from each other.  "Give me the red ball!" Lily shouted. "It's mine!"  "No, it's mine!" Tom yelled. "It's mine!"  They pulled so hard that they separate the red ball from each other. They both fell down and started to cry.  Mom heard the noise and came to see what was wrong. She saw the red ball on the floor and the tears on Lily's face. She was not angry, but she was sad.  "Why are you fighting?" Mom asked. "You both have the red ball, but you both have the same toy."  "That is boring!" Lily said. "You both need to share and take turns."  "Okay!" Mom said. "But you can both have the red ball

### Prompt: Lily loved going to the park because
- **Output:** Lily loved going to the park because it was always free. One day, she saw a boy crying and her knee started to hurt.  'Hello there, little girl,' the boy said, pointing to his knee.   'I'm sad because my knee hurts,' he said.  The man smiled and said, 'Don't worry, I have a band-aid in my bag and I'll take it out to make you feel better.'  The boy went back to his house and his mother put the band-aid on his knee.   The man smiled and said, 'I'm happy to see this band-aid and I'm glad so you can help me!'   The boy thanked his mother and helped the man get back in his shoes. The man said goodbye and the boy smiled as he skipped away.

---

## Long-form continuation
- **Settings:** temperature=0.8, top_k=50, max_tokens=100

### Prompt: The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from the 7th century BC, with selective stretches later joined together by Qin Shi Huang, the first emperor of China. Little of the Qin wall remains. Later on, many successive dynasties built and maintained multiple stretches of border walls. The best-known sections of the wall were built by the Ming dynasty.
- **Output:** The Great Wall of China is a series of fortifications that were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. Several walls were built from the 7th century BC, with selective stretches later joined together by Qin Shi Huang, the first emperor of China. Little of the Qin wall remains. Later on, many successive dynasties built and maintained multiple stretches of border walls. The best-known sections of the wall were built by the Ming dynasty. The Chinese Army was established in 1899, and became the first American Army occupied by the army. The city has a large population of 100,000 people; the entire city of China is the largest in the world. The city grew by 31 percent, with a total of 322,000 square feet. The most recent World War II military shrank the city's population, while the largest, and the largest, is the most populous city in China. In 1901,