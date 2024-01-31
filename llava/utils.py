import datetime
import logging
import logging.handlers
import os
import sys

import re
import requests
import base64
import time

from llava.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

handler = None


def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when='D', utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """
    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ''
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == '\n':
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != '':
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ''


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json",
               "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        flagged = False
    except KeyError as e:
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def postprocess_llava_output(text):
    pass


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


# Function to extract the generated attributes from LLMs (e.g., LLava, GPT-4V)
def extract_attributes(text):
    # Regular expression pattern
    pattern = r'(Required|Likely):([\s\S]*?)(?=(Required|Likely|$))'

    # Find all matches
    matches = re.findall(pattern, text)

    # Dictionary to store the extracted attributes
    attributes = {"Required": [], "Likely": []}

    # Iterate over matches and extract attributes
    for section, content, _ in matches:
        # Extract bullet points, considering both '*' and '-'
        bullet_points = re.findall(r'[\*\-] (.+)', content.strip())
        attributes[section].extend(bullet_points)

    return attributes["Required"], attributes["Likely"]

# Function to extract Answers: from LLM-generated outputs
def preprocess_gpt_output(response, start_str="Answer:"):
    answer_start_idx = response.find(start_str) + len(start_str) if response.find(start_str) != -1 else 0
    prepro_response = response[answer_start_idx:]
    return prepro_response 


def readCsv(path):
    '''
    The five fields of each line in `conceptnet-assertions-5.7.0.csv` are (tab-separated):

    - The URI of the whole edge
    - The relation expressed by the edge
    - The node at the start of the edge
    - The node at the end of the edge
    - A JSON structure of additional information about the edge, such as its weight

    '''
    i = 0
    if os.path.exists(path):
        print(f"Loading file from [{path}]")
        freader = open(path)
        text = freader.readline()
        while text != "":
            print(text.split("\t"))
            text = freader.readline()
            if i > 10:
                break
            i += 1
        freader.close()
        

def remove_text_after_triple_newline(text):
    """
    Removes any text in the input string that appears after the first occurrence of triple newline characters.

    :param text: The input text string.
    :return: The modified text with everything after the first triple newline removed.
    """
    # Find the first occurrence of triple newline
    index = text.find("\n\n\n")

    # If triple newline is found, slice the string up to that point; else return the original text
    return text[:index] if index != -1 else text


# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


def openai_gpt_call(client, system_prompt, user_prompt, 
                    model, image_path=None, max_tokens=256, 
                    temp=0.0, max_retries=5, retry_delay=2):
    retry_count = 0
    while True:
        try:
            if image_path is not None:
                base64_image = encode_image(image_path)  # base64-encoded image
                user_content = [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }}
                ]
            else:
                user_content = [{"type": "text", "text": user_prompt}]

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            response = client.chat.completions.create(
                model=model, 
                messages=messages,
                max_tokens=max_tokens,
                temperature=temp
            )

            # If response is successful, break the loop
            if response:
                return response, response.choices[0].message.content

        except Exception as e:
            # Handle specific exceptions if needed
            if "content_policy_violation" in e.message:
                # Note, this format may change due to OpenAI's API update
                '''

                [ GPT-4 raw_response sample 
                ] 
                "gpt4_raw_response": {"id": "chatcmpl-8ee6lDVpdQ8xt5JdWKjv0wBNl4XXY",
                  "choices": [{"finish_reason": "stop", "index": 0, 
                  "message": {"content": "Required:\n- elongated, slender body\n- smooth, shiny skin\n- lack of limbs\n- visible segmentation along the body\n\nLikely:\n- burrowing in soil\n- found in moist environments\n- presence of a prostomium (a lip-like extension over the mouth)", 
                  "role": "assistant", "function_call": null, "tool_calls": null}}], "created": 1704697987, "model": "gpt-4-1106-vision-preview", "object": "chat.completion", "system_fingerprint": null, "usage": {"completion_tokens": 58, "prompt_tokens": 454, "total_tokens": 512}}
                
                '''
                raw_response = {"id": "-1",
                                "choices": [{"finish_reason": "error", "index": 0, 
                                             "message": {"content": "None"}}],
                                "role": "assistant",
                                "function_call": None,
                                "tool_calls": None}
                return raw_response, raw_response["choices"][0]

        retry_count += 1
        if retry_count >= max_retries:
            raise Exception("Maximum number of retries reached. API call failed.")

        time.sleep(retry_delay)