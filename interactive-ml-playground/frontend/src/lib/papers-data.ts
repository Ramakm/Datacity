export interface PaperComponent {
  name: string;
  description: string;
  code: string;
}

export interface Paper {
  id: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string;
  link: string;
  category: "foundational" | "embeddings" | "rnn" | "transformer" | "llm";
  description: string;
  hasBreakdown: boolean;
  components?: PaperComponent[];
}

export const papers: Paper[] = [
  {
    id: "word2vec-2013",
    title: "Efficient Estimation of Word Representations in Vector Space",
    shortTitle: "Word2Vec",
    year: 2013,
    authors: "Mikolov et al.",
    link: "https://arxiv.org/abs/1301.3781",
    category: "embeddings",
    description: "Skip-gram and CBOW architectures for learning word embeddings.",
    hasBreakdown: true,
    components: [
      {
        name: "Vocabulary Building",
        description: "Build vocabulary from corpus and create word-to-index mappings.",
        code: `from collections import Counter

def build_vocab(corpus, min_freq=5):
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence.split())

    vocab = {w: i for i, (w, c) in enumerate(
        word_counts.items()) if c >= min_freq}
    vocab['<UNK>'] = len(vocab)
    return vocab

# Example
corpus = ["the cat sat on mat", "the dog ran fast"]
vocab = build_vocab(corpus, min_freq=1)`,
      },
      {
        name: "Skip-gram Data Preparation",
        description: "Generate (center, context) pairs for skip-gram training.",
        code: `import numpy as np

def generate_skipgram_pairs(sentence, vocab, window=2):
    words = sentence.split()
    pairs = []
    for i, center in enumerate(words):
        if center not in vocab:
            continue
        start = max(0, i - window)
        end = min(len(words), i + window + 1)
        for j in range(start, end):
            if i != j and words[j] in vocab:
                pairs.append((vocab[center], vocab[words[j]]))
    return pairs

# Generate training pairs
pairs = generate_skipgram_pairs("the cat sat on the mat", vocab)`,
      },
      {
        name: "Skip-gram Model",
        description: "Neural network with embedding and output layers for word prediction.",
        code: `import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center, context):
        center_emb = self.center_embeddings(center)
        context_emb = self.context_embeddings(context)
        score = torch.sum(center_emb * context_emb, dim=1)
        return score

model = SkipGram(vocab_size=len(vocab), embedding_dim=100)`,
      },
      {
        name: "Negative Sampling",
        description: "Efficient training by sampling negative examples instead of full softmax.",
        code: `import torch.nn.functional as F

def negative_sampling_loss(model, center, context, neg_samples):
    # Positive score
    pos_score = model(center, context)
    pos_loss = F.logsigmoid(pos_score)

    # Negative scores
    neg_score = model(center.repeat(len(neg_samples)), neg_samples)
    neg_loss = F.logsigmoid(-neg_score).sum()

    return -(pos_loss + neg_loss)

# Sample negative words based on frequency
def sample_negatives(vocab_size, k=5):
    return torch.randint(0, vocab_size, (k,))`,
      },
      {
        name: "CBOW Model",
        description: "Continuous Bag of Words - predict center word from context.",
        code: `class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_words):
        # context_words: (batch, context_size)
        embeds = self.embeddings(context_words)
        # Average context embeddings
        hidden = embeds.mean(dim=1)
        output = self.linear(hidden)
        return F.log_softmax(output, dim=1)

cbow = CBOW(vocab_size=len(vocab), embedding_dim=100)`,
      },
      {
        name: "Training Loop",
        description: "Complete training procedure with optimizer and loss tracking.",
        code: `def train_word2vec(model, pairs, epochs=10, lr=0.01):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for center, context in pairs:
            center_t = torch.tensor([center])
            context_t = torch.tensor([context])
            neg_samples = sample_negatives(len(vocab))

            optimizer.zero_grad()
            loss = negative_sampling_loss(
                model, center_t, context_t, neg_samples)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")`,
      },
    ],
  },
  {
    id: "lstm-1997",
    title: "Long Short-Term Memory",
    shortTitle: "LSTM",
    year: 1997,
    authors: "Hochreiter & Schmidhuber",
    link: "https://www.bioinf.jku.at/publications/older/2604.pdf",
    category: "rnn",
    description: "Gated recurrent architecture solving vanishing gradient problem.",
    hasBreakdown: true,
    components: [
      {
        name: "LSTM Cell Structure",
        description: "Core LSTM cell with forget, input, and output gates.",
        code: `import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Gates: forget, input, candidate, output
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat([x, h_prev], dim=1)

        f = torch.sigmoid(self.W_f(combined))  # Forget gate
        i = torch.sigmoid(self.W_i(combined))  # Input gate
        c_tilde = torch.tanh(self.W_c(combined))  # Candidate
        o = torch.sigmoid(self.W_o(combined))  # Output gate

        c = f * c_prev + i * c_tilde  # New cell state
        h = o * torch.tanh(c)  # New hidden state

        return h, c`,
      },
      {
        name: "Forget Gate",
        description: "Decides what information to discard from cell state.",
        code: `# Forget gate determines what to forget from previous cell state
# f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)

def forget_gate(x, h_prev, W_f, b_f):
    combined = torch.cat([h_prev, x], dim=-1)
    f_t = torch.sigmoid(combined @ W_f + b_f)
    return f_t

# When f_t ≈ 0: forget everything
# When f_t ≈ 1: remember everything
# Values in between: partial forgetting`,
      },
      {
        name: "Input Gate",
        description: "Decides what new information to store in cell state.",
        code: `# Input gate has two parts:
# 1. i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
# 2. c_tilde = tanh(W_c · [h_{t-1}, x_t] + b_c)

def input_gate(x, h_prev, W_i, W_c, b_i, b_c):
    combined = torch.cat([h_prev, x], dim=-1)

    # What values to update
    i_t = torch.sigmoid(combined @ W_i + b_i)

    # Candidate values to add
    c_tilde = torch.tanh(combined @ W_c + b_c)

    return i_t, c_tilde

# New information = i_t * c_tilde`,
      },
      {
        name: "Cell State Update",
        description: "Combine forget and input gates to update cell state.",
        code: `# Cell state update equation:
# C_t = f_t * C_{t-1} + i_t * c_tilde

def update_cell_state(c_prev, f_t, i_t, c_tilde):
    # Forget old information
    forgotten = f_t * c_prev

    # Add new information
    new_info = i_t * c_tilde

    # New cell state
    c_t = forgotten + new_info
    return c_t

# Cell state acts as a "conveyor belt"
# Information flows with minimal modification`,
      },
      {
        name: "Output Gate",
        description: "Decides what to output based on cell state.",
        code: `# Output gate:
# o_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o)
# h_t = o_t * tanh(C_t)

def output_gate(x, h_prev, c_t, W_o, b_o):
    combined = torch.cat([h_prev, x], dim=-1)

    # What parts of cell state to output
    o_t = torch.sigmoid(combined @ W_o + b_o)

    # Filter cell state through tanh and gate
    h_t = o_t * torch.tanh(c_t)

    return h_t

# h_t is the hidden state passed to next step
# Also used for predictions`,
      },
      {
        name: "Full LSTM Layer",
        description: "Process entire sequence through LSTM cells.",
        code: `class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                           num_layers, batch_first=True)

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size,
                            self.hidden_size)
            c0 = torch.zeros(self.num_layers, batch_size,
                            self.hidden_size)
            hidden = (h0, c0)

        output, (h_n, c_n) = self.lstm(x, hidden)
        return output, (h_n, c_n)

# Usage
lstm = LSTM(input_size=128, hidden_size=256)
x = torch.randn(32, 10, 128)  # batch, seq, features
out, hidden = lstm(x)`,
      },
    ],
  },
  {
    id: "attention-2017",
    title: "Attention Is All You Need",
    shortTitle: "Transformer",
    year: 2017,
    authors: "Vaswani et al.",
    link: "https://arxiv.org/abs/1706.03762",
    category: "transformer",
    description: "Self-attention mechanism replacing recurrence entirely.",
    hasBreakdown: true,
    components: [
      {
        name: "Tokenization",
        description: "Convert text into tokens using BPE, WordPiece, or SentencePiece.",
        code: `from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, world!", return_tensors="pt")
print(tokens)

# Manual BPE-style tokenization concept
def simple_tokenize(text, vocab):
    tokens = []
    for word in text.lower().split():
        if word in vocab:
            tokens.append(vocab[word])
        else:
            tokens.append(vocab['<UNK>'])
    return tokens`,
      },
      {
        name: "Embeddings",
        description: "Map token IDs to dense vector representations.",
        code: `import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        # Scale embeddings by sqrt(d_model)
        return self.embedding(x) * (self.d_model ** 0.5)

vocab_size = 30522
d_model = 768
embedding = TokenEmbedding(vocab_size, d_model)`,
      },
      {
        name: "Positional Encoding",
        description: "Add position information using sinusoidal functions.",
        code: `import torch
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]`,
      },
      {
        name: "Scaled Dot-Product Attention",
        description: "Core attention mechanism with Query, Key, Value.",
        code: `import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Apply mask (for decoder self-attention)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    # Softmax to get attention weights
    attn_weights = F.softmax(scores, dim=-1)

    # Apply attention to values
    output = torch.matmul(attn_weights, V)

    return output, attn_weights`,
      },
      {
        name: "Multi-Head Attention",
        description: "Parallel attention heads for different representation subspaces.",
        code: `class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # Linear projections and reshape for heads
        Q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_out, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(attn_out)`,
      },
      {
        name: "Feed Forward Network",
        description: "Position-wise fully connected layers with GELU activation.",
        code: `class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x`,
      },
      {
        name: "Layer Normalization",
        description: "Normalize across features for stable training.",
        code: `class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta`,
      },
      {
        name: "Encoder Block",
        description: "Complete transformer encoder layer with residual connections.",
        code: `class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x`,
      },
    ],
  },
  {
    id: "bert-2018",
    title: "BERT: Pre-training of Deep Bidirectional Transformers",
    shortTitle: "BERT",
    year: 2018,
    authors: "Devlin et al.",
    link: "https://arxiv.org/abs/1810.04805",
    category: "transformer",
    description: "Masked language modeling and next sentence prediction.",
    hasBreakdown: true,
    components: [
      {
        name: "BERT Tokenization",
        description: "WordPiece tokenization with special tokens [CLS], [SEP], [MASK].",
        code: `from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text = "Hello, how are you?"
tokens = tokenizer.tokenize(text)
print(tokens)  # ['hello', ',', 'how', 'are', 'you', '?']

# Encode with special tokens
encoded = tokenizer.encode_plus(
    text,
    add_special_tokens=True,
    max_length=512,
    padding='max_length',
    return_tensors='pt'
)
# [CLS] hello , how are you ? [SEP] [PAD] ...`,
      },
      {
        name: "BERT Embeddings",
        description: "Sum of token, segment, and position embeddings.",
        code: `class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_segments=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.seg_emb = nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device)

        embeddings = (
            self.token_emb(input_ids) +
            self.pos_emb(pos_ids) +
            self.seg_emb(segment_ids)
        )
        return self.dropout(self.norm(embeddings))`,
      },
      {
        name: "Masked Language Model (MLM)",
        description: "Predict randomly masked tokens in the input.",
        code: `import random

def mask_tokens(inputs, tokenizer, mlm_prob=0.15):
    labels = inputs.clone()

    # Create probability matrix
    prob_matrix = torch.full(labels.shape, mlm_prob)

    # Don't mask special tokens
    special_mask = tokenizer.get_special_tokens_mask(labels.tolist())
    prob_matrix.masked_fill_(torch.tensor(special_mask, dtype=torch.bool), 0.0)

    masked_indices = torch.bernoulli(prob_matrix).bool()
    labels[~masked_indices] = -100  # Only compute loss on masked

    # 80% [MASK], 10% random, 10% unchanged
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape)
    inputs[indices_random] = random_words[indices_random]

    return inputs, labels`,
      },
      {
        name: "Next Sentence Prediction (NSP)",
        description: "Binary classification if sentence B follows sentence A.",
        code: `class NextSentencePrediction(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, pooled_output):
        # pooled_output: [CLS] token representation
        return self.classifier(pooled_output)

def create_nsp_data(sentences, tokenizer):
    pairs = []
    for i in range(len(sentences) - 1):
        # 50% actual next sentence
        if random.random() > 0.5:
            pairs.append((sentences[i], sentences[i+1], 1))
        # 50% random sentence
        else:
            rand_idx = random.randint(0, len(sentences)-1)
            pairs.append((sentences[i], sentences[rand_idx], 0))
    return pairs`,
      },
      {
        name: "BERT Model Architecture",
        description: "Stack of transformer encoder blocks with pooler.",
        code: `class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 n_heads=12, d_ff=3072, max_len=512):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len)

        self.encoder_layers = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        # Pooler for [CLS] token
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh()
        )

    def forward(self, input_ids, segment_ids, attention_mask=None):
        x = self.embeddings(input_ids, segment_ids)

        for layer in self.encoder_layers:
            x = layer(x, attention_mask)

        # Pool [CLS] token
        pooled = self.pooler(x[:, 0])

        return x, pooled`,
      },
      {
        name: "Fine-tuning for Classification",
        description: "Add classification head on top of BERT for downstream tasks.",
        code: `from transformers import BertForSequenceClassification

class BertClassifier(nn.Module):
    def __init__(self, n_classes, pretrained='bert-base-uncased'):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained, num_labels=n_classes
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

# Training
model = BertClassifier(n_classes=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)`,
      },
    ],
  },
  {
    id: "gpt1-2018",
    title: "Improving Language Understanding by Generative Pre-Training",
    shortTitle: "GPT-1",
    year: 2018,
    authors: "Radford et al.",
    link: "https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf",
    category: "llm",
    description: "Generative pre-training for language understanding tasks.",
    hasBreakdown: true,
    components: [
      {
        name: "Causal Self-Attention",
        description: "Masked attention that only attends to previous tokens.",
        code: `def causal_attention_mask(seq_len):
    # Lower triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=1024):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        # Register causal mask
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len)
        )

    def forward(self, x):
        seq_len = x.size(1)
        mask = self.mask[:, :, :seq_len, :seq_len]
        return self.attn(x, x, x, mask)`,
      },
      {
        name: "GPT Decoder Block",
        description: "Transformer decoder block with causal attention.",
        code: `class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x`,
      },
      {
        name: "Language Model Head",
        description: "Project hidden states to vocabulary for next token prediction.",
        code: `class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12,
                 n_heads=12, d_ff=3072, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device)

        x = self.token_emb(input_ids) + self.pos_emb(pos_ids)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits`,
      },
      {
        name: "Autoregressive Training",
        description: "Train to predict the next token given previous context.",
        code: `def compute_lm_loss(logits, labels):
    # Shift so we predict next token
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss

# Training loop
model = GPT(vocab_size=50257)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for batch in dataloader:
    input_ids = batch['input_ids']
    logits = model(input_ids)
    loss = compute_lm_loss(logits, input_ids)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()`,
      },
      {
        name: "Text Generation",
        description: "Generate text autoregressively with sampling strategies.",
        code: `def generate(model, tokenizer, prompt, max_len=100, temperature=1.0, top_k=50):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt')

    for _ in range(max_len):
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[:, -1, :] / temperature

            # Top-k sampling
            top_k_logits, top_k_indices = torch.topk(next_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            next_token_idx = torch.multinomial(probs, 1)
            next_token = top_k_indices.gather(-1, next_token_idx)

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            if next_token.item() == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0])`,
      },
    ],
  },
  // Papers without breakdown
  {
    id: "shannon-1948",
    title: "A Mathematical Theory of Communication",
    shortTitle: "Information Theory",
    year: 1948,
    authors: "Claude Shannon",
    link: "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
    category: "foundational",
    description: "Foundation of information theory, entropy, and communication systems.",
    hasBreakdown: false,
  },
  {
    id: "bengio-2003",
    title: "A Neural Probabilistic Language Model",
    shortTitle: "Neural LM",
    year: 2003,
    authors: "Bengio et al.",
    link: "https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf",
    category: "foundational",
    description: "Introduced neural network-based language modeling and word embeddings.",
    hasBreakdown: false,
  },
  {
    id: "glove-2014",
    title: "GloVe: Global Vectors for Word Representation",
    shortTitle: "GloVe",
    year: 2014,
    authors: "Pennington et al.",
    link: "https://nlp.stanford.edu/pubs/glove.pdf",
    category: "embeddings",
    description: "Co-occurrence matrix factorization for word vectors.",
    hasBreakdown: false,
  },
  {
    id: "fasttext-2017",
    title: "Enriching Word Vectors with Subword Information",
    shortTitle: "FastText",
    year: 2017,
    authors: "Bojanowski et al.",
    link: "https://arxiv.org/abs/1607.04606",
    category: "embeddings",
    description: "Character n-gram based word embeddings for morphologically rich languages.",
    hasBreakdown: false,
  },
  {
    id: "turing-1950",
    title: "Computing Machinery and Intelligence",
    shortTitle: "Turing Test",
    year: 1950,
    authors: "Alan Turing",
    link: "https://redirect.cs.umbc.edu/courses/471/papers/turing.pdf",
    category: "foundational",
    description: "The imitation game and foundations of AI philosophy.",
    hasBreakdown: false,
  },
];

export const categoryLabels: Record<string, string> = {
  foundational: "FOUNDATIONAL",
  embeddings: "EMBEDDINGS",
  rnn: "RNN / LSTM",
  transformer: "TRANSFORMER",
  llm: "LARGE LANGUAGE MODELS",
};

export const categoryColors: Record<string, string> = {
  foundational: "border-terminal-accent text-terminal-accent",
  embeddings: "border-blue-500 text-blue-500",
  rnn: "border-purple-500 text-purple-500",
  transformer: "border-terminal-warning text-terminal-warning",
  llm: "border-red-500 text-red-500",
};

export function getPaperById(id: string): Paper | undefined {
  return papers.find(p => p.id === id);
}

export function getPapersWithBreakdown(): Paper[] {
  return papers.filter(p => p.hasBreakdown);
}
