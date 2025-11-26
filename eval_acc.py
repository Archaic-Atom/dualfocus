import os
import openai
import argparse
import json
import ast
import yaml
from multiprocessing.pool import Pool
from tqdm import tqdm
import time


API_KEY_WORKER = None
API_BASE_WORKER = None
GPT_MODEL_WORKER = "gpt-3.5-turbo"


def parse_cli_args():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using GPT.")
    parser.add_argument(
        "--config",
        type=str,
        default="eval_acc_config.yaml",
        help="Path to the evaluation configuration YAML file."
    )
    return parser.parse_args()


def init_worker(api_key, api_base, gpt_model):
    global API_KEY_WORKER, API_BASE_WORKER, GPT_MODEL_WORKER
    API_KEY_WORKER = api_key
    API_BASE_WORKER = api_base
    GPT_MODEL_WORKER = gpt_model

    openai.api_key = API_KEY_WORKER
    if API_BASE_WORKER:
        openai.api_base = API_BASE_WORKER


def annotate_batch(batch_data_for_gpt, output_dir_gpt):
    """
    Sends a batch of QA pairs to GPT for evaluation.
    Each item in batch_data_for_gpt is a tuple (key, qa_set_dict).
    """

    results = []
    for key, qa_set in batch_data_for_gpt:
        question = qa_set['question']
        answer = qa_set['answer']
        pred = qa_set['pred']

        if not pred:
            pred = "No prediction provided."
        if answer == "N/A" or not answer:
            print(f"Skipping sample {key} due to missing ground truth answer.")
            response_dict = {"pred": "N/A", "score": 0}
        else:
            try:
                completion = openai.chat.completions.create(
                    model=GPT_MODEL_WORKER,
                    messages=[
                        {
                            "role": "system",
                            "content":
                                "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                                "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                                "------"
                                "##INSTRUCTIONS: "
                                "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                                "- Consider synonyms or paraphrases as valid matches.\n"
                                "- Evaluate the correctness of the prediction compared to the answer."
                        },
                        {
                            "role": "user",
                            "content":
                                "Please evaluate the following video-based question-answer pair:\n\n"
                                f"Question: {question}\n"
                                f"Correct Answer: {answer}\n"
                                f"Predicted Answer: {pred}\n\n"
                                "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                                "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                                "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                                "For example, your response should look like this: {'pred': 'yes', 'score': 4}."
                        }
                    ]
                )
                response_message = completion.choices[0].message.content
                try:
                    response_dict = ast.literal_eval(response_message.strip())
                    if not isinstance(response_dict,
                                      dict) or 'pred' not in response_dict or 'score' not in response_dict:
                        print(
                            f"Warning: GPT response for {key} malformed (structure): {response_message}. Defaulting score.")
                        response_dict = {"pred": "error", "score": 0}

                    current_score = response_dict.get('score')
                    if isinstance(current_score, (float, str)):
                        try:
                            response_dict['score'] = int(round(float(current_score)))
                        except ValueError:
                            print(
                                f"Warning: GPT response for {key} malformed score value: {current_score}. Defaulting score.")
                            response_dict['score'] = 0
                    elif not isinstance(current_score, int):
                        print(
                            f"Warning: GPT response for {key} has unexpected score type: {type(current_score)}, value: {current_score}. Defaulting score.")
                        response_dict['score'] = 0


                except (SyntaxError, ValueError) as e:
                    print(
                        f"Warning: GPT response for {key} malformed (ast eval): {response_message}. Error: {e}. Defaulting score.")
                    response_dict = {"pred": "error", "score": 0}

            except openai.APIError as e:
                print(f"OpenAI API error for {key}: {e}. Defaulting score.")
                response_dict = {"pred": "api_error", "score": 0}
            except openai.OpenAIError as e:
                print(f"General OpenAI error for {key}: {e}. Defaulting score.")
                response_dict = {"pred": "openai_error", "score": 0}
            except Exception as e:
                print(f"Unexpected error processing GPT for {key}: {e}. Defaulting score.")
                response_dict = {"pred": "exception", "score": 0}

        with open(os.path.join(output_dir_gpt, f"{key}.json"), "w", encoding='utf-8') as f:
            json.dump([response_dict, qa_set], f, ensure_ascii=False, indent=2)
        results.append((key, [response_dict, qa_set]))
    return results


def main_eval_acc(config_path):
    global API_KEY, API_BASE, GPT_MODEL

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    acc_config = config['accuracy_evaluator_settings']
    base_output_dir = config['output_base_dir']

    # 从配置文件或环境变量加载主进程的API配置
    API_KEY = os.environ.get("OPENAI_API_KEY", acc_config.get('openai_api_key'))
    API_BASE = os.environ.get("OPENAI_API_BASE", acc_config.get('openai_api_base'))
    GPT_MODEL = acc_config.get('gpt_model', "gpt-3.5-turbo")

    if not API_KEY or "YOUR_OPENAI_API_KEY" in API_KEY:
        print(
            "Error: OpenAI API Key not set. Please set it in eval_config2.yaml or as an environment variable OPENAI_API_KEY.")
        return

    predictions_input_path = os.path.join(base_output_dir, "generated_preds", acc_config['predictions_file'])
    gpt_output_dir = os.path.join(base_output_dir, acc_config['gpt_output_subdir'])
    gpt_summary_path = os.path.join(base_output_dir, acc_config['gpt_summary_filename'])

    os.makedirs(gpt_output_dir, exist_ok=True)

    if not os.path.exists(predictions_input_path):
        print(f"Error: Predictions file not found at {predictions_input_path}")
        return

    # --- MODIFICATION START ---
    eval_max_len = acc_config.get('eval_max_len', -1)
    if not isinstance(eval_max_len, int):
        print(f"Warning: eval_max_len ('{eval_max_len}') in config is not an integer. Defaulting to load all (-1).")
        eval_max_len = -1

    pred_contents = []
    if eval_max_len == 0:
        print("eval_max_len is 0. No predictions will be loaded.")
    else:
        with open(predictions_input_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if eval_max_len > 0 and i >= eval_max_len:
                    print(
                        f"Limiting evaluation to the first {eval_max_len} predictions as specified by 'eval_max_len'.")
                    break
                pred_contents.append(json.loads(line.strip()))

    print(f"Loaded {len(pred_contents)} predictions to process from {predictions_input_path}.")
    # --- MODIFICATION END ---

    if not acc_config.get('use_gpt_eval', True):
        print("GPT evaluation is disabled. Implement simple metrics if needed.")
        return

    tasks_for_gpt = []
    for i, sample in enumerate(pred_contents):
        key = f"{sample.get('video_id', 'unk_vid')}_{sample.get('question_id', f'q_{i}')}"
        tasks_for_gpt.append((key, sample))

    num_gpt_tasks = acc_config.get('num_gpt_tasks', 1)
    all_results_map = {}

    completed_files_names = {f[:-5] for f in os.listdir(gpt_output_dir) if f.endswith(".json")}
    incomplete_tasks = []
    for key, qa_set in tasks_for_gpt:
        if key not in completed_files_names:
            incomplete_tasks.append((key, qa_set))

    if not incomplete_tasks:
        print("All samples already evaluated by GPT (found corresponding .json files). Loading existing results.")
    else:
        print(f"Found {len(incomplete_tasks)} samples to evaluate with GPT.")
        chunk_size = (len(incomplete_tasks) + num_gpt_tasks - 1) // num_gpt_tasks
        task_chunks = [incomplete_tasks[i:i + chunk_size] for i in range(0, len(incomplete_tasks), chunk_size)]
        starmap_args = [(chunk, gpt_output_dir) for chunk in task_chunks]

        with Pool(processes=num_gpt_tasks,
                  initializer=init_worker,
                  initargs=(API_KEY, API_BASE, GPT_MODEL)) as pool:
            worker_outputs = list(
                tqdm(pool.starmap(annotate_batch, starmap_args), total=len(task_chunks), desc="GPT Evaluation Batches"))

        for batch_result in worker_outputs:
            for key, result_data in batch_result:
                all_results_map[key] = result_data
        print(f"GPT evaluation completed for {len(all_results_map)} new samples.")

    final_combined_contents = {}
    for file_name in os.listdir(gpt_output_dir):
        if file_name.endswith(".json"):
            file_key = file_name[:-5]
            file_path = os.path.join(gpt_output_dir, file_name)
            try:
                with open(file_path, "r", encoding='utf-8') as json_file:
                    content = json.load(json_file)
                    final_combined_contents[file_key] = content
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    with open(gpt_summary_path, "w", encoding='utf-8') as f:
        json.dump(final_combined_contents, f, ensure_ascii=False, indent=2)
    print(f"GPT evaluation summary saved to: {gpt_summary_path}")

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    error_eval_count = 0

    for key, result_pair in tqdm(final_combined_contents.items(), desc="Calculating Stats"):
        if not isinstance(result_pair, list) or len(result_pair) < 1:
            print(f"Skipping malformed result for key {key}: {result_pair}")
            error_eval_count += 1
            continue

        gpt_eval_result = result_pair[0]
        if not isinstance(gpt_eval_result, dict):
            print(f"Skipping malformed GPT eval dict for key {key}: {gpt_eval_result}")
            error_eval_count += 1
            continue

        try:
            score_val = gpt_eval_result.get('score', 0)
            if isinstance(score_val, (int)):
                score = score_val
            elif isinstance(score_val, (float, str)):
                try:
                    score = int(round(float(score_val)))
                except ValueError:
                    print(f"Error converting score to int for key {key}, data: {gpt_eval_result}. Defaulting to 0.")
                    score = 0
            else:
                print(f"Unexpected score type for key {key}, data: {gpt_eval_result}. Defaulting to 0.")
                score = 0

            score_sum += score
            count += 1

            pred_match_str = gpt_eval_result.get('pred', 'error')
            if "yes" in pred_match_str.lower():
                yes_count += 1
            elif "no" in pred_match_str.lower():
                no_count += 1
            else:
                error_eval_count += 1

        except (ValueError, TypeError) as e:
            print(f"Error processing score for key {key}, data: {gpt_eval_result}. Error: {e}")
            error_eval_count += 1
            time.sleep(60)

    if count > 0:
        valid_accuracy_evals = yes_count + no_count
        accuracy = (yes_count / valid_accuracy_evals * 100) if valid_accuracy_evals > 0 else 0.0
        average_score = score_sum / count

        print("\n--- GPT Evaluation Summary ---")
        print(f"Total samples where GPT response was processed: {count}")
        print(f"Meaningful Match ('yes'): {yes_count}")
        print(f"No Meaningful Match ('no'): {no_count}")
        print(
            f"Evaluation Errors/Skipped/Malformed Pred: {error_eval_count}")
        if valid_accuracy_evals > 0:
            print(f"Accuracy (Yes / (Yes + No)): {accuracy:.2f}%")
        else:
            print("Accuracy: N/A (no valid 'yes' or 'no' evaluations)")
        print(f"Average Score (0-5, over {count} processed samples): {average_score:.2f}")
    else:
        print("No results to calculate statistics.")


if __name__ == "__main__":
    API_KEY = None
    API_BASE = None
    GPT_MODEL = "gpt-3.5-turbo"

    cli_args = parse_cli_args()
    main_eval_acc(cli_args.config)