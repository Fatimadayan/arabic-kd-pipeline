# ═══════════════════════════════════════════════════════════════
# ACAI — Complete Fine-Tuning Guide for Your Friend
# + Image Generation Roadmap
# + Audio/Voice Roadmap
# ═══════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────
# PART 1: WHAT TO TELL YOUR FRIEND (KD Fine-Tuning)
# ────────────────────────────────────────────────────────────────

FRIEND_BRIEFING = """
=== ACAI Fine-Tuning — Complete Briefing for Your Friend ===

YOUR DATASET STATUS:
• 4,927 samples (2428 Arabic + 2499 English) from 32B teacher model ✅
• This is a solid dataset for fine-tuning a 7B model
• Target: Qwen2.5-7B-Instruct (student model)

WHAT YOU NEED TO DO (exact steps):

STEP 1 — Verify the data is ready:
    wc -l /data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl
    # Should show ~4927

STEP 2 — Check data quality (run this):
    python3 -c "
    import json
    with open('/data/datasets/user151/qwen-arabic-kd/data/teacher_32b_responses.jsonl') as f:
        lines = [json.loads(l) for l in f if l.strip()]
    print(f'Total: {len(lines)}')
    # Check format
    print('Sample:', json.dumps(lines[0], ensure_ascii=False)[:200])
    # Check average length
    avg = sum(len(l.get('output','') or l.get('response','')) for l in lines) / len(lines)
    print(f'Avg response length: {avg:.0f} chars')
    "

STEP 3 — Update SLURM script to use 32B data:
    sed -i 's|teacher_responses.jsonl|teacher_32b_responses.jsonl|' \
        /data/datasets/user151/qwen-arabic-kd/slurm/train_sft_7b.slurm
    
    # Also increase max_seq_length for richer 32B responses:
    sed -i 's|max_seq_length=1024|max_seq_length=2048|' \
        /data/datasets/user151/qwen-arabic-kd/slurm/train_sft_7b.slurm

STEP 4 — Submit the training job:
    sbatch /data/datasets/user151/qwen-arabic-kd/slurm/train_sft_7b.slurm
    squeue -u user151  # Monitor

STEP 5 — Expected results:
    • Training time: ~3-4 hours on 2x A100
    • Final eval loss: should be < 1.5
    • Checkpoint saved every epoch

STEP 6 — After training, test the model:
    cd /data/datasets/user151/qwen-arabic-kd
    python scripts/test_model.py --model output/checkpoint-final \
        --prompt "شلونك يا خوي؟" \
        --prompt "What is the CBB's role in Bahrain?" \
        --prompt "اشرح مفهوم التعلم الآلي"

STEP 7 — Export to Ollama (to use in ACAI):
    # Convert to GGUF format
    git clone https://github.com/ggerganov/llama.cpp /data/datasets/$USER/llama.cpp
    python /data/datasets/$USER/llama.cpp/convert_hf_to_gguf.py \
        /data/datasets/user151/qwen-arabic-kd/output/checkpoint-final \
        --outtype q4_k_m \
        --outfile /data/datasets/user151/acai/models/acai-qwen7b-v1.gguf
    
    # Create Modelfile
    cat > /data/datasets/user151/acai/Modelfile << 'EOF'
FROM /data/datasets/user151/acai/models/acai-qwen7b-v1.gguf
SYSTEM "أنت ACAI — مساعد ذكي متخصص في العربية والخليج."
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF
    
    # Load into Ollama
    ollama create acai-qwen7b:v1 -f /data/datasets/user151/acai/Modelfile
    
    # Test
    ollama run acai-qwen7b:v1 "شلونك يا خوي؟"

STEP 8 — Update ACAI backend to use fine-tuned model:
    # Edit backend/.env on your Windows laptop:
    # SPECIALIST_MODEL=acai-qwen7b:v1
    # Restart backend: uvicorn main:app --port 8000 --reload

=== WHAT TO EXPECT ===
Your fine-tuned model will be better than base Qwen2.5-7B at:
1. Arabic dialect understanding (especially Gulf/Bahraini)
2. GCC-specific knowledge (CBB, SAMA, Vision 2030)
3. Arabic-English code-switching
4. Cultural context of the Arab world

The 32B teacher model transfers its deeper Arabic knowledge
into the smaller 7B student model — that's the magic of KD!
"""

# ────────────────────────────────────────────────────────────────
# PART 2: PROMPTS TO SEND TO OTHER AIs FOR MORE KD DATA
# ────────────────────────────────────────────────────────────────

KD_PROMPTS_FOR_GPT4 = """
=== SEND THIS TO GPT-4 / CLAUDE / GEMINI ===

Generate 100 high-quality training samples for a Bahraini Arabic AI model.

Each sample must be in this EXACT JSON format:
{"instruction": "...", "input": "", "output": "...", "language": "ar"}

Requirements:
- 30 samples: Bahraini dialect daily conversations (use: الحين وايد حيل زين شلون مب صج تره)
- 25 samples: CBB/banking Q&A in Arabic (accurate, cite CBB Rulebook)
- 20 samples: Bahraini to MSA dialect translation
- 15 samples: Arabic AI/tech explanations with natural English code-switching
- 10 samples: GCC policy questions in English with detailed English answers

Quality rules:
- Arabic outputs must be AUTHENTIC, not Google Translate Arabic
- Banking answers must be ACCURATE (cite CBB Rulebook modules)
- Dialect must sound like REAL Bahrainis, not formal Arabic
- Length: 50-300 words per output

Return a valid JSON array of exactly 100 objects.
"""

# ────────────────────────────────────────────────────────────────
# PART 3: IMAGE GENERATION — SHOULD WE ADD IT?
# ────────────────────────────────────────────────────────────────

IMAGE_GENERATION_GUIDE = """
=== IMAGE GENERATION IN ACAI ===

YES — you should add it. Here's exactly how:

OPTION A: DALL-E 3 via Anthropic (easiest, best quality)
─────────────────────────────────────────────────────────
Add a 7th agent called "مُصوِّر" (Musawwir — the Image Maker)

In the frontend, when this agent is active:
- Send the Arabic/English prompt to OpenAI DALL-E 3 API
- Display the generated image in the chat
- Arabic prompts get auto-translated to English for DALL-E

Cost: ~$0.04 per image (very affordable)
API: api.openai.com/v1/images/generations

OPTION B: Stable Diffusion (free, runs on your A100)
─────────────────────────────────────────────────────
On Hayrat lab:
  pip install diffusers transformers
  python -c "
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained('stabilityai/stable-diffusion-2-1')
  pipe = pipe.to('cuda')
  image = pipe('A beautiful Arabic calligraphy of بسم الله').images[0]
  image.save('output.png')
  "

Then add a FastAPI endpoint in backend/main.py:
  @app.post('/api/generate-image')
  async def generate_image(prompt: str):
      # Call Stable Diffusion
      return {"image_url": "..."}

OPTION C: Arabic-aware Image Generation
────────────────────────────────────────
For Arabic text in images (calligraphy, etc.):
Use ArabiGAN or ArabicTextRenderer
These handle Arabic script correctly in images

MY RECOMMENDATION:
Start with DALL-E 3 (Option A) — 2 hours to implement.
Then add local Stable Diffusion for the research paper.

IMPLEMENTATION FOR ACAI:
Add this agent to AGENTS array in App_v5.jsx:
{
  id: "musawwir",
  name: "مُصوِّر",
  nameEn: "Image Creator",
  title: "عميل توليد الصور",
  icon: "◐",
  color: "#EC4899",
  badge: "IMAGE GEN",
  placeholder: "صِف الصورة التي تريد إنشاءها...",
}

Then in the agent handler, call DALL-E instead of text generation.
"""

# ────────────────────────────────────────────────────────────────
# PART 4: AUDIO / VOICE — SHOULD WE ADD IT?
# ────────────────────────────────────────────────────────────────

AUDIO_VOICE_GUIDE = """
=== AUDIO & VOICE IN ACAI ===

YES — this is your BIGGEST differentiator.
No Arabic AI currently has good Bahraini dialect voice.

THREE COMPONENTS:

1. SPEECH TO TEXT (User speaks Arabic → text)
─────────────────────────────────────────────
Use: OpenAI Whisper (best Arabic ASR, free, open source)

On your A100 lab:
  pip install openai-whisper
  whisper --model large-v3 --language Arabic audio.mp3

In the ACAI frontend:
  // Browser microphone → Whisper API
  navigator.mediaDevices.getUserMedia({ audio: true })
  // Send audio blob to backend
  // Backend runs Whisper → returns Arabic text
  // Text goes to the selected agent

2. TEXT TO SPEECH (Agent response → voice)
───────────────────────────────────────────
Option A: ElevenLabs Arabic (best quality, $)
Option B: Azure Cognitive Services Arabic TTS (good, $)
Option C: XTTS-v2 (free, runs on A100, good Arabic)

For Bahraini dialect voice:
  pip install TTS
  tts --text "الحين أجي وايد زين" --model_name "tts_models/multilingual/multi-dataset/xtts_v2"

3. BAHRAINI VOICE CLONING (your unique contribution!)
──────────────────────────────────────────────────────
Record 30 minutes of a Bahraini native speaker.
Fine-tune XTTS-v2 on those recordings.
Now your agent literally speaks in Bahraini dialect voice.

THIS IS PUBLISHABLE — no one has a Bahraini dialect TTS model.

IMPLEMENTATION PRIORITY:
Week 1: Add Whisper STT to frontend (mic button in chat)
Week 2: Add XTTS-v2 TTS (play button on responses)
Week 3: Bahraini voice fine-tuning on A100

=== WHICH TO DO FIRST? ===

For your mid-test submission:
→ Focus on App_v5.jsx UI + backend improvements

For Phase 3:
→ Add STT/TTS after the paper draft is done

For the startup:
→ Voice interface is the #1 feature enterprises want
→ "The first AI that speaks in Bahraini Arabic" = major differentiator
"""

# ────────────────────────────────────────────────────────────────
# PART 5: IMPROVED باحث (Search Agent)
# ────────────────────────────────────────────────────────────────

BAHITH_IMPROVEMENTS = """
=== HOW باحث BEATS PERPLEXITY ===

WHAT PERPLEXITY DOES:
Query → Single web search → Summarize results

WHAT ACAI باحث DOES (with web_search_20250305):
Query → Analyze → Multiple targeted searches (Arabic + English)
→ Verify sources → Cross-reference → Synthesize
→ Present with citations + confidence score

KEY DIFFERENCES:
1. Searches in Arabic AND English for GCC topics
2. Prioritizes official sources (CBB, government sites)
3. Shows search queries being executed in real-time
4. Confidence score per source
5. Explicit "I don't know" when no reliable source found

THE باحث AGENT IN App_v5.jsx:
- Uses Anthropic's web_search_20250305 tool directly
- Shows animated search chips as it searches
- Requires Anthropic API key (entered once per session)
- Falls back to backend if key not available

TO MAKE IT EVEN FASTER:
- The speed is limited by how fast web search results come back
- Anthropic's search is usually 2-5 seconds per query
- Total response: 5-15 seconds (similar to Perplexity)

NO HALLUCINATIONS because:
- Every claim must come from an actual search result
- System prompt explicitly says "NEVER make up sources"
- If search returns nothing relevant, agent says so honestly
"""

if __name__ == "__main__":
    print(FRIEND_BRIEFING)
    print("\n" + "="*60)
    print(IMAGE_GENERATION_GUIDE)
    print("\n" + "="*60)
    print(AUDIO_VOICE_GUIDE)
