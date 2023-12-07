# Importing the necessary libraries:
# This code imports the T5 model and tokenizer from the transformers library. These are essential for working with the T5 model

from transformers import T5ForConditionalGeneration, T5Tokenizer


# Load the pre-trained T5 model & tokenizer
# Here, the code loads the pre-trained T5 model (T5ForConditionalGeneration) using the from_pretrained method. 
# The model you're using is 'ramsrigouthamg/t5_squad_v1'.
# Similarly, the code initializes the T5 tokenizer (T5Tokenizer) using the from_pretrained method. 
# The tokenizer uses the 't5-base' model and has a maximum sequence length of 1024 tokens.

from transformers import T5ForConditionalGeneration, T5Tokenizer
question_model = T5ForConditionalGeneration.from_pretrained('priya_t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)

# Defining the get_question_from_answer function:
# This function takes an answer and its corresponding context as inputs. 
# It then creates a string text that concatenates the context and answer with the necessary formatting.

def get_question_from_answer(answer, context):
    text = f"context: {context} answer: {answer} </s>"

    # Tokenize the input text using the tokenizer
    # The code tokenizes the text using the tokenizer's encode_plus method. 
    # It sets the max_length to 256 and enables padding and truncation. 
    # The truncation_strategy parameter determines how the text is truncated if it exceeds the max_length. The encoded input is returned as PyTorc sensors.
    
    max_len = 256
    encoding = question_tokenizer.encode_plus(
        text,
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True,
        truncation_strategy="longest_first",
        return_tensors="pt"
    )
    
    
   # Generate the question using the T5 model
   # this code prepares the input tensors (input_ids and attention_mask) from the encoding. 
   # Then, it passes these tensors to the T5 model's generate method to generate questions. 
   # Several parameters are provided, such as num_beams (the number of beams for beam search), 
   # no_repeat_ngram_size (prevents repetition of n-grams), and max_length (maximum length of the generated question).
    
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    outs = question_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        early_stopping=True,
        num_beams=5,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        max_length=200
    )

    # Decode and format the question
    # The code decodes the generated question tensors (outs) using the tokenizer's decode method. 
    # It removes the prefix "question:" and leading/trailing whitespace from the decoded text. 
    # The resulting question is returned by the function.
    
    dec = [question_tokenizer.decode(ids) for ids in outs]
    question = dec[0].replace("question:", "").strip()
    return question



import json
# Reading a JSON file
# This code reads a JSON file containing interview schema data. 
# The file is opened and its contents are loaded into the data variable.

file_path = "C:\\Users\\91999\\Downloads\\interviewerSchema.json.json"

# Read the JSON file
with open(file_path, "r") as file:
    data = json.load(file)
data


# Check if Python, Django, NumPy, and scikit-learn (sklearn) skills are present in the schema
# these lines check if the skills "python," "django," "numpy," and "sklearn" are present in the JSON schema.
# It iterates through the "Techskills" list in the schema and checks if the corresponding skill is present using the any() function.

python_skill_present = any(skill.get("python") for skill in data.get("skills", {}).get("Techskills", []))
django_skill_present = any(skill.get("django") for skill in data.get("skills", {}).get("Techskills", []))
numpy_skill_present = any(skill.get("numpy") for skill in data.get("skills", {}).get("Techskills", []))
sklearn_skill_present = any(skill.get("sklearn") for skill in data.get("skills", {}).get("Techskills", []))


# Generate Python questions
# Generating Python, Django, NumPy, and scikit-learn questions:
# Based on the skills present in the schema, the code generates questions for each skill using the get_question_from_answer() function and prints the generated question along with the original answer.

if python_skill_present:
    
    python_answers = [
        
        "In Python, the self keyword is used as a convention to refer to the instance of a class within the class itself.",
        "Decorators in python are a way to modify or enhance the behavior of functions or classes without directly modifying their source code.",
        "PEP 8 is a style guide for Python code and PEP 8 specifically focuses on the style conventions for writing Python code.",
        "Python uses private heap space to manage memory.", 
        "In Python, inheritance is a mechanism that allows a class to inherit properties (attributes and methods) from another class."
    ]
    
    for answer in python_answers:
        generated_question = get_question_from_answer(answer, "Python")
        generated_question = generated_question.replace("<pad> ", "").replace("</s>", "")
        print(f"Question: {generated_question}")
        print(f"Answer: {answer}")
        print()


# Generate Django questions

if django_skill_present:
    django_answers = [
        
        "Django provides built-in support for internationalization (i18n) and localization (l10n), it includes features for translating text, handling date and number formatting based on different locales, and switching between multiple languages in the application",
        "The Django admin site is an automatically generated administration interface that allows authorized users to manage the application's data models.",
        "Django follows a slightly modified version of the MVC architectural pattern, known as Model-View-Template (MVT). Models represent the data structure, views handle the business logic and interaction with models, and templates handle the presentation layer.",
        "Django middleware is a component that sits between the web server and the view function, allowing you to process requests and responses globally.",
        "Django's template system allows developers to separate the presentation logic from the Python code."
    
    ]
    
    for answer in django_answers:
        generated_question = get_question_from_answer(answer, "Django")
        generated_question = generated_question.replace("<pad> ", "").replace("</s>", "")
        print(f"Question: {generated_question}")
        print(f"Answer: {answer}")
        print()


# Generate NumPy questions

if numpy_skill_present:  
    numpy_answers = [
        
        "NumPy stands for Numerical Python and it is a powerful library in Python for numerical computations, Its main purpose is to provide efficient and convenient handling of large arrays and matrices of numerical data, along with a collection of mathematical functions to operate on these arrays.",
        "A NumPy array is a grid of values, all of the same data type, and indexed by a tuple of non-negative integers. It is the fundamental data structure in NumPy and provides several advantages over regular Python lists, including faster execution of operations, optimized memory usage, and a wide range of mathematical functions and operations specifically designed for arrays.",
        "A one-dimensional array in NumPy is a simple list of values, while a multi-dimensional array represents a table of elements with rows and columns, similar to a matrix. Multi-dimensional arrays can have any number of dimensions, such as 2D for matrices or 3D for representing volumes of data.",
        "You can reshape or resize a NumPy array using the np.reshape() function or by directly modifying the shape attribute of the array. For example: new_arr = np.reshape(arr, (2, 3)) or arr.shape = (2, 3)",
        "NumPy provides several functions to generate arrays with specific patterns or values, such as np.zeros(), np.ones(), np.arange(), np.linspace(), np.random.rand(), np.random.randint(), and np.eye()."
    
    ]
    for answer in numpy_answers:
        generated_question = get_question_from_answer(answer, "NumPy")
        generated_question = generated_question.replace("<pad> ", "").replace("</s>", "")
        print(f"Question: {generated_question}")
        print(f"Answer: {answer}")
        print()

# Generate scikit-learn (sklearn) questions

if sklearn_skill_present:
    sklearn_answers = [
        
        "sklearn provides various modules and components, including datasets for loading and exploring datasets, preprocessing for data preprocessing techniques, model_selection for model selection and evaluation, metrics for performance evaluation metrics, estimators for machine learning algorithms, and ensemble for ensemble methods, among others.",
        "Estimator objects in sklearn are used to fit models to data and make predictions. They encapsulate the learning algorithms and have methods such as fit() to train the model, predict() to make predictions, and score() to evaluate the model's performance.",
        "sklearn provides a datasets module that offers various datasets for practice and experimentation. You can use functions like load_iris(), load_digits(), or fetch_california_housing() to load datasets. Once loaded, you can explore the dataset attributes, such as data, target, feature_names, and target_names.",
        "sklearn offers several preprocessing techniques, including scaling features with StandardScaler or MinMaxScaler, handling missing values with SimpleImputer, encoding categorical variables with OneHotEncoder or LabelEncoder, and reducing dimensionality with techniques like Principal Component Analysis (PCA).",
        "The Pipeline module in scikit-learn allows you to chain multiple preprocessing steps and machine learning models into a single object. It helps in organizing and automating the workflow, enabling easier model building, training, and evaluation."
    
     ]
    
    for answer in sklearn_answers:
        generated_question = get_question_from_answer(answer, "scikit-learn")
        generated_question = generated_question.replace("<pad> ", "").replace("</s>", "")
        print(f"Question: {generated_question}")
        print(f"Answer: {answer}")
        print()
        
# If none of the skills are present in the schema, this message is printed to indicate that no questions can be generated.

else:
    print("No Python, django, numpy and sklearn tech skill found in the JSON schema.")

# Overall, the code reads a JSON schema, checks for specific skills, and generates questions based on those skills using a pre-trained T5 model for question generation. 
    