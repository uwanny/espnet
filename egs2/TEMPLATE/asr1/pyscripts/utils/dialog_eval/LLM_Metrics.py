from multiprocessing import Pool
from typing import List

import numpy as np
import torch
from pyscripts.utils.dialog_eval.vert import (
    get_auto_bleu2_geometric,
    get_self_bleu2_geometric,
    run_f,
)
from scipy.stats import gmean
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def perplexity(LLM_Output: str, model_id: str = "gpt2") -> str:
    """
    Compute the perplexity of the given text using a specified model from the
    `evaluate` library (default: GPT-2).

    Args:
        LLM_Output str:
            The text (string) for which perplexity is to be computed.
        model_id (str, optional):
            The identifier of the model to use for computing
            perplexity. Defaults to "gpt2".

    Returns:
        str:
            A formatted string showing the perplexity of the
            provided text(s), for example:
            "Perplexity: 45.23\n"

    Raises:
        ImportError:
            If the `evaluate` library is not installed or cannot be imported.

    Example:
        >>> text = "Hello world, this is a test."
        >>> result = perplexity(text, model_id="gpt2")
        >>> print(result)
        "Perplexity: 27.34\n"
    """
    try:
        import evaluate
    except Exception as e:
        print("Error: evaluate is not properly installed.")
        raise e
    perplexity = evaluate.load("perplexity", module_type="metric")
    results = perplexity.compute(model_id=model_id, predictions=[LLM_Output])
    return f"Perplexity: {results['mean_perplexity']:.2f}\n"


def vert(LLM_response_arr: List[str]) -> str:
    """
    Calculate and return Self BLEU-2, Auto BLEU-2 and VERT-2
    metrics for a list of LLM responses.

    Args:
        LLM_response_arr (List[str]):
            A list of responses (strings) generated by the language
            model acting as text dialog response generator.

    Returns:
        str:
            A formatted string that includes each computed metric and the final
            VERT value, for example:

            "Self-BLEU2-geometric: 42.13
             Auto-BLEU2-geometric: 38.94
             VERT: 40.5
             "

    Example:
        >>> # Suppose we have the following LLM responses:
        >>> responses = ["Hello world", "Foo bar", "Lorem ipsum dolor sit amet"]
        >>> result = vert(responses)
        >>> print(result)
        "Self-BLEU2-geometric: 42.13
         Auto-BLEU2-geometric: 38.94
         VERT: 40.5
         "
    """
    terms = [x.strip().split() for x in LLM_response_arr]

    tasks = [
        ("Self-BLEU2-geometric", get_self_bleu2_geometric),
        ("Auto-BLEU2-geometric", get_auto_bleu2_geometric),
    ]
    n_processes = min(16, len(tasks))
    with Pool(n_processes) as pool:
        metrics = pool.map(run_f, [(t[1], terms) for t in tasks])
    metric_arr = []
    str1 = ""
    for (metric_name, _), metric in zip(tasks, metrics):
        metric, sem = np.mean(metric), np.std(metric) / np.sqrt(len(metric))

        metric, sem = [round(100 * x, 2) for x in [metric, sem]]
        metric_arr.append(metric)

        str1 += f"{metric_name}: {metric}\n"
    str1 += f"VERT: {round(gmean(metric_arr), 2)}\n"
    return str1


def bert_score(
    total_response_arr: List[str], bert_model_name: str = "bert-base-uncased"
) -> str:
    """
    Compute a cosine similarity score between the concatenated
    context (all but the last element)
    and the final response (last element) using a BERT-based model.
    This serves as a simplified
    measure of how closely the response aligns with the preceding context semantically.

    Args:
        total_response_arr (List[str]):
            A list of strings. The last element represents the response,
            while all other elements
            are treated as the context.
        bert_model_name (str, optional):
            The name or path of the BERT model to use (from the Hugging Face Model Hub).
            Defaults to "bert-base-uncased".

    Returns:
        str:
            A string containing the cosine similarity
            (as a percentage) followed by a newline.
            For example:
                "Cosine Similarity: 85.67\n"

    Example:
        >>> total_responses = [
        ...     "User: Hi, how are you?",
        ...     "Assistant: I'm good! How can I help you today?",
        ...     "User: Can you tell me a joke?",
        ...     "Assistant: Sure! Here's one: Why did the chicken join a band?"
        ... ]
        >>> result = bert_score(total_responses, bert_model_name="bert-base-uncased")
        >>> print(result)
        "Cosine Similarity: 75.89\n"
    """

    def cosine_similarity_context_response(context, response, model, tokenizer):
        # Tokenize and encode both context and response
        context_inputs = tokenizer(context, return_tensors="pt", truncation=True)
        response_inputs = tokenizer(response, return_tensors="pt", truncation=True)
        for k in context_inputs:
            context_inputs[k] = context_inputs[k].cuda()
        for k in response_inputs:
            response_inputs[k] = response_inputs[k].cuda()

        # Get embeddings from the model
        with torch.no_grad():
            context_embedding = model(**context_inputs).last_hidden_state.mean(dim=1)
            response_embedding = model(**response_inputs).last_hidden_state.mean(dim=1)

        # Compute cosine similarity
        similarity = cosine_similarity(
            context_embedding.cpu().numpy(), response_embedding.cpu().numpy()
        )
        return similarity[0][0]

    bert_model = AutoModel.from_pretrained(bert_model_name).cuda()
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    similarity = cosine_similarity_context_response(
        " ".join(total_response_arr[:-1]),
        total_response_arr[-1],
        bert_model,
        bert_tokenizer,
    )
    return f"Cosine Similarity: {similarity*100:.2f}" + "\n"


def DialoGPT_perplexity(
    user_utterance: str,
    response: str,
    dialog_model_name: str = "microsoft/DialoGPT-medium",
) -> str:
    """
    Compute the perplexity of a response given a user utterance using a pre-trained
    DialoGPT model. The function loads DialoGPT (medium by default)
    from the Hugging Face Model Hub, then calculates the perplexity
    for the
    (context + response) sequence.

    Args:
        user_utterance (str):
            The user utterance preceding the model's response.
        response (str):
            The generated response whose perplexity needs to be evaluated.

    Returns:
        str:
            A formatted string containing the DialoGPT perplexity score. For example:
            "DialoGPT Perplexity: 25.67\n"

    Example:
        >>> user_text = "Hi, how are you today?"
        >>> system_response = "I'm good, thank you! How can I help you?"
        >>> result = DialoGPT_perplexity(user_text, system_response)
        >>> print(result)
        "DialoGPT Perplexity: 31.45\n"
    """

    def evaluate_response_with_dialoGPT(context, response, model, tokenizer):
        """
        Evaluate the appropriateness of a response based on the
        given context using DialoGPT.

        Args:
            context (str): The dialogue context (previous conversation).
            response (str): The generated response to evaluate.
            model: Pre-trained DialoGPT model.
            tokenizer: Corresponding tokenizer for the DialoGPT model.

        Returns:
            float: Perplexity score of the response given the context.
        """
        model.eval()

        # Combine context and response as input
        input_text = context + tokenizer.eos_token + response + tokenizer.eos_token
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
        inputs["input_ids"] = inputs["input_ids"].cuda()
        inputs["attention_mask"] = inputs["attention_mask"].cuda()
        # import pdb;pdb.set_trace()

        # Compute model outputs and loss
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"].cuda())
            loss = outputs.loss

        # Calculate perplexity
        perplexity = torch.exp(loss)
        return perplexity.cpu().item()

    # Load DialoGPT model and tokenizer
    model_name = dialog_model_name
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    perplexity = evaluate_response_with_dialoGPT(
        user_utterance, response, model, tokenizer
    )
    return f"DialoGPT Perplexity: {perplexity:.2f}" + "\n"
