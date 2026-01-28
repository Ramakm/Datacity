"use client";

import { useState } from "react";
import { ArrowLeft, FileText, Code, ChevronRight, ExternalLink, BookOpen } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";

interface Paper {
  id: string;
  title: string;
  shortTitle: string;
  year: number;
  authors: string;
  link: string;
  category: "foundational" | "embeddings" | "rnn" | "transformer" | "llm";
  description: string;
}

interface PaperComponent {
  name: string;
  description: string;
  code: string;
}

const papers: Paper[] = [
  {
    id: "shannon-1948",
    title: "A Mathematical Theory of Communication",
    shortTitle: "Information Theory",
    year: 1948,
    authors: "Claude Shannon",
    link: "https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf",
    category: "foundational",
    description: "Foundation of information theory, entropy, and communication systems.",
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
  },
  {
    id: "word2vec-2013",
    title: "Efficient Estimation of Word Representations in Vector Space",
    shortTitle: "Word2Vec",
    year: 2013,
    authors: "Mikolov et al.",
    link: "https://arxiv.org/abs/1301.3781",
    category: "embeddings",
    description: "Skip-gram and CBOW architectures for learning word embeddings.",
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
  },
  {
    id: "cnn-text-2014",
    title: "A Convolutional Neural Network for Modelling Sentences",
    shortTitle: "CNN for NLP",
    year: 2014,
    authors: "Kalchbrenner et al.",
    link: "https://arxiv.org/abs/1404.2188",
    category: "foundational",
    description: "Dynamic convolutional neural network for sentence modeling.",
  },
  {
    id: "backprop-1986",
    title: "Learning Internal Representations by Error Propagation",
    shortTitle: "Backpropagation (RNN)",
    year: 1986,
    authors: "Rumelhart et al.",
    link: "https://www.cs.toronto.edu/~hinton/absps/naturebp.pdf",
    category: "rnn",
    description: "Backpropagation algorithm enabling deep network training.",
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
  },
  {
    id: "colah-lstm",
    title: "Understanding LSTM Networks",
    shortTitle: "LSTM Explained (Colah)",
    year: 2015,
    authors: "Christopher Olah",
    link: "https://colah.github.io/posts/2015-08-Understanding-LSTMs/",
    category: "rnn",
    description: "Visual explanation of LSTM architecture and gates.",
  },
  {
    id: "elmo-2018",
    title: "Deep Contextualized Word Embeddings",
    shortTitle: "ELMo",
    year: 2018,
    authors: "Peters et al.",
    link: "https://arxiv.org/abs/1802.05365",
    category: "rnn",
    description: "Contextualized embeddings from bidirectional LSTM language model.",
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
  },
  {
    id: "gpt2-2019",
    title: "Language Models Are Unsupervised Multitask Learners",
    shortTitle: "GPT-2",
    year: 2019,
    authors: "Radford et al.",
    link: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
    category: "llm",
    description: "Scaling language models for zero-shot task transfer.",
  },
  {
    id: "gpt3-2020",
    title: "Language Models Are Few-Shot Learners",
    shortTitle: "GPT-3",
    year: 2020,
    authors: "Brown et al.",
    link: "https://arxiv.org/abs/2005.14165",
    category: "llm",
    description: "In-context learning and emergent abilities at scale.",
  },
  {
    id: "sentence-bert-2019",
    title: "Sentence-BERT: Sentence Embeddings using Siamese Networks",
    shortTitle: "Sentence-BERT",
    year: 2019,
    authors: "Reimers & Gurevych",
    link: "https://arxiv.org/abs/1908.10084",
    category: "transformer",
    description: "Efficient sentence embeddings for semantic similarity.",
  },
  {
    id: "rlhf-2022",
    title: "Training Language Models to Follow Instructions with Human Feedback",
    shortTitle: "InstructGPT (RLHF)",
    year: 2022,
    authors: "Ouyang et al.",
    link: "https://arxiv.org/abs/2203.02155",
    category: "llm",
    description: "Reinforcement learning from human feedback for alignment.",
  },
  {
    id: "llama2-2023",
    title: "Llama 2: Open Foundation and Fine-Tuned Chat Models",
    shortTitle: "LLaMA 2",
    year: 2023,
    authors: "Touvron et al.",
    link: "https://arxiv.org/abs/2307.09288",
    category: "llm",
    description: "Open-source large language model with safety fine-tuning.",
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
  },
];

const transformerComponents: PaperComponent[] = [
  {
    name: "Tokenization",
    description: "Convert text into tokens (subwords) using BPE, WordPiece, or SentencePiece algorithms.",
    code: `from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("Hello, world!", return_tensors="pt")
print(tokens)`,
  },
  {
    name: "Embeddings",
    description: "Map token IDs to dense vector representations in a learned embedding space.",
    code: `import torch.nn as nn

vocab_size = 30522
d_model = 768
embedding = nn.Embedding(vocab_size, d_model)
embedded = embedding(tokens["input_ids"])`,
  },
  {
    name: "Positional Encoding",
    description: "Add position information using sinusoidal functions or learned embeddings.",
    code: `import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe`,
  },
  {
    name: "Multi-Head Attention",
    description: "Parallel attention heads computing scaled dot-product attention on Q, K, V projections.",
    code: `import torch.nn.functional as F

def attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn = F.softmax(scores, dim=-1)
    return torch.matmul(attn, V)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)`,
  },
  {
    name: "Layer Normalization",
    description: "Normalize activations across features for stable training and faster convergence.",
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
    name: "Feed Forward",
    description: "Two-layer MLP with GELU/ReLU activation applied position-wise.",
    code: `class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.gelu(self.linear1(x))))`,
  },
  {
    name: "Output Projection",
    description: "Project hidden states to vocabulary logits for next token prediction.",
    code: `class OutputProjection(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        logits = self.proj(x)
        return F.log_softmax(logits, dim=-1)`,
  },
];

const categoryLabels: Record<string, string> = {
  foundational: "FOUNDATIONAL",
  embeddings: "EMBEDDINGS",
  rnn: "RNN / LSTM",
  transformer: "TRANSFORMER",
  llm: "LARGE LANGUAGE MODELS",
};

const categoryColors: Record<string, string> = {
  foundational: "border-terminal-accent text-terminal-accent",
  embeddings: "border-blue-500 text-blue-500",
  rnn: "border-purple-500 text-purple-500",
  transformer: "border-terminal-warning text-terminal-warning",
  llm: "border-red-500 text-red-500",
};

export default function ResearchPapersPage() {
  const [selectedComponent, setSelectedComponent] = useState<PaperComponent>(transformerComponents[0]);
  const [filterCategory, setFilterCategory] = useState<string>("all");

  const filteredPapers = filterCategory === "all"
    ? papers
    : papers.filter(p => p.category === filterCategory);

  const categories = ["all", "foundational", "embeddings", "rnn", "transformer", "llm"];

  return (
    <div className="min-h-screen bg-terminal-bg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-2 font-mono text-xs uppercase tracking-terminal text-terminal-black hover:text-terminal-accent transition-colors mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            RETURN TO TERMINAL
          </Link>

          <div className="bg-terminal-panel border-2 border-terminal-black p-6">
            <div className="flex items-start justify-between">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-terminal-black flex items-center justify-center">
                  <FileText className="w-6 h-6 text-terminal-mint" />
                </div>
                <div>
                  <h1 className="heading-terminal text-2xl md:text-3xl text-terminal-black mb-2">
                    RESEARCH_PAPERS
                  </h1>
                  <p className="font-mono text-xs text-terminal-black/70 max-w-2xl leading-relaxed">
                    CURATED COLLECTION OF FOUNDATIONAL ML/NLP PAPERS //
                    FROM INFORMATION THEORY TO LARGE LANGUAGE MODELS
                  </p>
                </div>
              </div>
              <div className="hidden sm:flex items-center gap-2">
                <span className="text-xs font-mono font-bold px-2 py-1 border-2 border-terminal-accent text-terminal-accent uppercase tracking-terminal">
                  {papers.length} PAPERS
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Category Filter */}
        <div className="mb-6 flex flex-wrap gap-2">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setFilterCategory(cat)}
              className={clsx(
                "px-3 py-1 font-mono text-xs uppercase tracking-terminal border-2 transition-all",
                filterCategory === cat
                  ? "bg-terminal-black text-terminal-mint border-terminal-black"
                  : "bg-transparent text-terminal-black border-terminal-black/30 hover:border-terminal-black"
              )}
            >
              {cat === "all" ? "ALL" : categoryLabels[cat]}
            </button>
          ))}
        </div>

        {/* Papers List */}
        <div className="bg-terminal-panel border-2 border-terminal-black mb-8">
          <div className="border-b-2 border-terminal-black p-4 flex items-center gap-3">
            <BookOpen className="w-5 h-5 text-terminal-black" />
            <h2 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black">
              PAPER_REGISTRY // {filteredPapers.length} ENTRIES
            </h2>
          </div>

          <div className="divide-y-2 divide-terminal-black/20 max-h-[500px] overflow-y-auto">
            {filteredPapers.map((paper) => (
              <a
                key={paper.id}
                href={paper.link}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-start gap-4 p-4 hover:bg-terminal-black/5 transition-colors group"
              >
                <div className="flex-shrink-0 w-12 text-center">
                  <span className="font-mono text-xs text-terminal-black/50">{paper.year}</span>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-start justify-between gap-4">
                    <div>
                      <h3 className="font-mono text-sm font-bold text-terminal-black group-hover:text-terminal-accent transition-colors">
                        {paper.title}
                      </h3>
                      <p className="font-mono text-xs text-terminal-black/60 mt-1">
                        {paper.authors}
                      </p>
                      <p className="font-mono text-xs text-terminal-black/50 mt-1">
                        {paper.description}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 flex-shrink-0">
                      <span className={clsx(
                        "text-xs font-mono font-bold px-2 py-0.5 border uppercase tracking-terminal",
                        categoryColors[paper.category]
                      )}>
                        {paper.shortTitle}
                      </span>
                      <ExternalLink className="w-4 h-4 text-terminal-black/30 group-hover:text-terminal-accent transition-colors" />
                    </div>
                  </div>
                </div>
              </a>
            ))}
          </div>
        </div>

        {/* Implementation Section */}
        <div className="bg-terminal-panel border-2 border-terminal-black">
          <div className="border-b-2 border-terminal-black p-4 flex items-center gap-3">
            <Code className="w-5 h-5 text-terminal-black" />
            <h2 className="font-mono font-bold text-sm uppercase tracking-terminal text-terminal-black">
              ATTENTION_IS_ALL_YOU_NEED // IMPLEMENTATION_BREAKDOWN
            </h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 divide-y-2 lg:divide-y-0 lg:divide-x-2 divide-terminal-black">
            {/* Component List */}
            <div className="p-4">
              <h3 className="font-mono text-xs font-bold uppercase tracking-terminal text-terminal-black/50 mb-4">
                TRANSFORMER COMPONENTS
              </h3>
              <div className="space-y-2">
                {transformerComponents.map((component, index) => (
                  <button
                    key={component.name}
                    onClick={() => setSelectedComponent(component)}
                    className={clsx(
                      "w-full flex items-center gap-3 p-3 font-mono text-left transition-all border-2",
                      selectedComponent.name === component.name
                        ? "bg-terminal-black text-terminal-mint border-terminal-black"
                        : "bg-transparent text-terminal-black border-terminal-black/20 hover:border-terminal-black"
                    )}
                  >
                    <span className="text-xs opacity-50">{String(index + 1).padStart(2, '0')}</span>
                    <span className="font-bold text-sm uppercase tracking-terminal flex-1">
                      {component.name}
                    </span>
                    <ChevronRight className={clsx(
                      "w-4 h-4 transition-transform",
                      selectedComponent.name === component.name ? "rotate-90" : ""
                    )} />
                  </button>
                ))}
              </div>
            </div>

            {/* Code Display */}
            <div className="p-4 bg-terminal-black/5">
              <div className="mb-4">
                <h3 className="font-mono text-sm font-bold uppercase tracking-terminal text-terminal-black mb-2">
                  {selectedComponent.name}
                </h3>
                <p className="font-mono text-xs text-terminal-black/70 leading-relaxed">
                  {selectedComponent.description}
                </p>
              </div>

              <div className="bg-terminal-black p-4 overflow-x-auto">
                <div className="flex items-center gap-2 mb-3 pb-2 border-b border-terminal-mint/30">
                  <div className="w-2 h-2 bg-red-500" />
                  <div className="w-2 h-2 bg-terminal-warning" />
                  <div className="w-2 h-2 bg-terminal-accent" />
                  <span className="ml-2 font-mono text-xs text-terminal-mint/60">
                    {selectedComponent.name.toLowerCase().replace(/ /g, '_')}.py
                  </span>
                </div>
                <pre className="font-mono text-xs text-terminal-mint whitespace-pre-wrap leading-relaxed">
                  {selectedComponent.code}
                </pre>
              </div>

              <div className="mt-4 p-3 border-2 border-dashed border-terminal-black/30">
                <p className="font-mono text-xs text-terminal-black/60">
                  <span className="font-bold text-terminal-accent">TIP:</span> Run this code in a Jupyter notebook
                  or Python environment with PyTorch installed.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Reading Order */}
        <div className="mt-8 bg-terminal-black text-terminal-mint p-6 border-2 border-terminal-black">
          <div className="flex items-center gap-2 mb-4 pb-3 border-b border-terminal-mint/30">
            <div className="w-3 h-3 bg-red-500" />
            <div className="w-3 h-3 bg-terminal-warning" />
            <div className="w-3 h-3 bg-terminal-accent" />
            <span className="ml-4 font-mono text-xs opacity-60">RECOMMENDED_READING_ORDER.md</span>
          </div>

          <div className="font-mono text-sm space-y-2">
            <p><span className="text-terminal-accent">$</span> cat ./reading_path.txt</p>
            <p className="opacity-70 mt-2">SUGGESTED PROGRESSION:</p>
            <p className="opacity-70">1. Information Theory (Shannon) → Neural LM (Bengio)</p>
            <p className="opacity-70">2. Word2Vec → GloVe → FastText</p>
            <p className="opacity-70">3. LSTM → ELMo → Attention</p>
            <p className="opacity-70">4. Transformer → BERT → GPT Series</p>
            <p className="opacity-70">5. RLHF → Modern LLMs</p>
            <p className="mt-3"><span className="text-terminal-accent">$</span> _<span className="animate-pulse">|</span></p>
          </div>
        </div>
      </div>
    </div>
  );
}
