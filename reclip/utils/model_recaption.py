from ollama import chat

def recaption_with_model(tag_list, model="qwen3"):
    # Convert tag list to prompt
    tags = ", ".join(tag_list) if isinstance(tag_list, list) else str(tag_list)
    user_prompt = f"Rewrite the following icon tags into a full, natural language description: {tags}"
    try:
        conversation = [{"role": "user", "content": user_prompt}]
        reply = chat(model=model, messages=conversation)
        return reply.message.content
    except Exception as e:
        return f"Qwen error: {e}"
