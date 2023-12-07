from flask import Flask, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

app = Flask(__name__)

# Load the pre-trained T5 model & tokenizer
question_model = T5ForConditionalGeneration.from_pretrained('priya_t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=1024)

# Defining the get_question_from_answer function
def get_question_from_answer(answer, context):
    text = f"context: {context} answer: {answer} </s>"
    max_len = 256
    encoding = question_tokenizer.encode_plus(
        text,
        max_length=max_len,
        pad_to_max_length=True,
        truncation=True,
        truncation_strategy="longest_first",
        return_tensors="pt"
    )
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
    dec = [question_tokenizer.decode(ids) for ids in outs]
    question = dec[0].replace("question:", "").strip()
    return question

# Main route for serving questions and answers
@app.route('/')
def index():
    # Reading a JSON file
    file_path = "C:\\Users\\91999\\Downloads\\interviewerSchema.json.json"

    # Read the JSON file
    with open(file_path, "r") as file:
        data = json.load(file)

    # Check if Python, Django, NumPy, and scikit-learn (sklearn) skills are present in the schema
    python_skill_present = any(skill.get("python") for skill in data.get("skills", {}).get("Techskills", []))
    django_skill_present = any(skill.get("django") for skill in data.get("skills", {}).get("Techskills", []))
    numpy_skill_present = any(skill.get("numpy") for skill in data.get("skills", {}).get("Techskills", []))
    sklearn_skill_present = any(skill.get("sklearn") for skill in data.get("skills", {}).get("Techskills", []))

    # Generate questions and answers
    questions_and_answers = []

    # Generate Python questions
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
            questions_and_answers.append({"question": generated_question, "answer": answer})

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
            questions_and_answers.append({"question": generated_question, "answer": answer})

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
            questions_and_answers.append({"question": generated_question, "answer": answer})

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
            questions_and_answers.append({"question": generated_question, "answer": answer})

    # If none of the skills are present in the schema
    else:
        return "No Python, Django, NumPy, and scikit-learn tech skill found in the JSON schema."

    # Render the questions and answers as JSON
    return jsonify(questions_and_answers)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

