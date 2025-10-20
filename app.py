from flask import Flask, request, render_template;
import pickle;
import pandas as pd;
import numpy as np;
import nltk;
import string as st
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('wordnet')
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import spacy;
nlp = spacy.load("en_core_web_sm")



app = Flask(__name__)

model = pickle.load(open("C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\DIC final\\models\\model.pkl",'rb'))
model_test = pickle.load(open("C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\DIC final\\models\\model_test.pkl",'rb'))
@app.route('/get_graph_image')

def get_graph():
    
    df = pd.read_csv("C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\DIC final\\src\\fake_job_postings _P2.csv")
    df.duplicated().any()
    df.drop_duplicates(inplace = True)    
    dropping_columns = ['salary_range','department'] #as the above two columns had more null values, we are dropping them
    df = df.drop(dropping_columns, axis=1)
    
    # Extract country from the 'location' column and create a new 'Country' column
    """df['countries'] = df['location'].apply(lambda x: x.split(',')[0].strip() if isinstance(x, str) else x)
    features = ['description', 'requirements', 'benefits']
    df[features] = df[features].fillna("No info") 
    df['employment_type'] = df['employment_type'].fillna("Other") 
    # Now, let's check the DataFrame to ensure the 'Country' column is created correctly
    df.drop(columns=['location'], inplace=False)
    df = df.apply(fill_missing_education_experience, axis=1)
    # printing the List of textual columns to be combined
    text_data = ['title', 'countries', 'required_education', 'required_experience', 'description', 'requirements', 'employment_type', 'benefits']

    # Creating a new column 'combined_text' by concatenating values from all specified columns
    df['text_feature'] = ''
    for col in text_data:
        df['text_feature'] += df[col].fillna('').astype(str) + ' '
    
    df['text_feature'] = df['text_feature'].str.lower()
    #removing the punctuation.
    df['text_feature'] = df['text_feature'].apply(lambda x: x.translate(str.maketrans('', '', st.punctuation)))
    df['text_feature'] = df['text_feature'].apply(tokening_removStopwords)
    df['text_feature'] = df['text_feature'].apply(lemmatize_text)"""

    """vectorizer = TfidfVectorizer()
    tfidf_vector = vectorizer.fit_transform(df['text_feature'])"""

    df['aggregate_feature'] = df[['has_questions', 'has_company_logo', 'telecommuting']].astype(str).agg(''.join, axis=1)
    
    #other_features = df[['has_company_logo', 'has_questions', 'telecommuting']].values

    # Combine TF-IDF transformed textual features with other features
    #X = np.concatenate((tfidf_vector.toarray(), other_features), axis=1)
    #output = model.predict();
    
    plot_g1(df['aggregate_feature'],df);
    
    

def plot_g1(df1,df):
    
    # Map the combined strings to unique categories
    category_map = {
        '000': 'No Questions, No Logo, No Telecommuting',
        '001': 'No Questions, No Logo, Telecommuting',
        '010': 'No Questions, Has Logo, No Telecommuting',
        '011': 'No Questions, Has Logo, Telecommuting',
        '100': 'Has Questions, No Logo, No Telecommuting',
        '101': 'Has Questions, No Logo, Telecommuting',
        '110': 'Has Questions, Has Logo, No Telecommuting',
        '111': 'Has Questions, Has Logo, Telecommuting'
    }
    df1 = df1.map(category_map)

    # Plot the aggregate feature against the output feature
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='aggregate_feature', hue='fraudulent')
    plt.xlabel('Aggregate Feature')
    plt.ylabel('Count')
    plt.title('Aggregate Feature vs. Fraudulent')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Fraudulent', loc='upper right')
    plt.tight_layout()

    plt.savefig('C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\flask\\static\\graph1.jpg');
    
    # Plotting the distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='employment_type', hue='fraudulent')
    plt.title('Distribution of Fraudulent and Non-Fraudulent Job Postings by Employment Type')
    plt.xlabel('Employment Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.legend(title='Fraudulent', loc='upper right')
    plt.tight_layout()
    plt.savefig('C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\flask\\static\\graph2.jpg');
    
    top_8_industries = df['industry'].value_counts().nlargest(8)

    plt.figure(figsize=(12, 6))
    top_8_industries.plot(kind='bar', stacked=True)
    plt.title('Top 8 Industries: Distribution of Fraudulent and Non-Fraudulent Job Postings')
    plt.xlabel('Industry')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Adjust rotation angle and alignment
    plt.legend(title='Fraudulent', loc='upper right')
    plt.tight_layout()
    plt.savefig('C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\flask\\static\\graph3.jpg');
    
    # Assuming 'fraudulent' column is available in the original DataFrame 'df'
    # Calculate the counts of fraudulent and non-fraudulent job postings for each industry
    industry_counts = df.groupby(['industry', 'fraudulent']).size().unstack()

    # Fill NaN values with 0 (if any)
    industry_counts = industry_counts.fillna(0)

    # Calculate the fraud rate for each industry
    industry_counts['Fraud Rate'] = industry_counts[1] / industry_counts.sum(axis=1)

    # Sort industries by fraud rate in descending order
    industry_counts = industry_counts.sort_values(by='Fraud Rate', ascending=False)

    # Select the top 10 industries with the highest fraud rate
    top_10_fraudulent_industries = industry_counts.head(10)

    # Plot the distribution of fraudulent and non-fraudulent job postings for the top 10 fraudulent industries
    plt.figure(figsize=(12, 6))
    top_10_fraudulent_industries.drop(columns='Fraud Rate').plot(kind='bar', stacked=True)
    plt.title('Top 10 Industries with Highest Fraud Rate: Distribution of Fraudulent and Non-Fraudulent Job Postings')
    plt.xlabel('Industry')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')  # Adjust rotation angle and alignment
    plt.legend(title='Fraudulent', loc='upper right')
    plt.tight_layout()
    plt.savefig('C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\flask\\static\\graph4.jpg');

def extract_education_and_experience_from_text(text):
    doc = nlp(text) # type: ignore

    # Extract specific education information if available
    education_entities = [ent.text.lower() for ent in doc.ents if 'education' in ent.text.lower()]
    required_education = education_entities[0] if education_entities else 'Not specified'

    # Extract specific experience information if available
    experience_entities = [ent.text.lower() for ent in doc.ents if 'experience' in ent.text.lower()]
    required_experience = experience_entities[0] if experience_entities else 'Not specified'

    return required_education, required_experience

# Apply the function to fill missing values in the DataFrame
def fill_missing_education_experience(row):
    if pd.isnull(row['required_education']) or pd.isnull(row['required_experience']):
        requirements_text = row['requirements']
        required_education, required_experience = extract_education_and_experience_from_text(requirements_text)
        
        # Populate missing values only if they are missing
        if pd.isnull(row['required_education']):
            row['required_education'] = required_education
        if pd.isnull(row['required_experience']):
            row['required_experience'] = required_experience
    
    return row

def tokening_removStopwords(text):
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word in stop_words]
    return ' '.join(filtered_tokens)  # Joining tokens back into a string with spaces

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)  # Tokenize the text
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize each token
    return ' '.join(lemmatized_tokens)  # Join the tokens back into a string

def analyze(title, countries, education, experience, description, logo, requirements, employment_type, has_questions, benefits, telecommuting):
    
    data = pd.read_csv(csv_file_path)
    # Create a dictionary with the provided variables
    data = {
        'title': [title],
        'countries': [countries],
        'required_education': [education],
        'required_experience': [experience],
        'description': [description],
        'has_company_logo': [logo],
        'requirements': [requirements],
        'employment_type': [employment_type],
        'has_questions':[has_questions],
        'benefits': [benefits],
        'telecommuting': [telecommuting]
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    df['has_company_logo'] = df['has_company_logo'].astype('int64')
    df['has_questions'] = df['has_questions'].astype('int64')
    df['telecommuting'] = df['telecommuting'].astype('int64')
    
    
    features = ['title', 'description', 'requirements']
    df[features] = df[features].fillna("No info")
    
    df = df.apply(fill_missing_education_experience, axis=1)
    
    # printing the List of textual columns to be combined
    text_data = ['title', 'countries', 'required_education', 'required_experience', 'description', 'requirements', 'employment_type', 'benefits']

    # Creating a new column 'combined_text' by concatenating values from all specified columns
    df['text_feature'] = ''
    for col in text_data:
        df['text_feature'] += df[col].fillna('').astype(str) + ' '
        
    # we are now cleaning the textual data.
    #print(df['text_feature'])
    # converting to lower case letters.
    df['text_feature'] = df['text_feature'].str.lower()
    #print(df['text_feature'])
    #removing the punctuation.
    df['text_feature'] = df['text_feature'].apply(lambda x: x.translate(str.maketrans('', '', st.punctuation)))
    #print(df['text_feature'])
    df['text_feature'] = df['text_feature'].apply(tokening_removStopwords)
    #print(df['text_feature'])
    df['text_feature'] = df['text_feature'].apply(lemmatize_text)
    

    # Initialize the TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    tfidf_vector = vectorizer.fit_transform(df['text_feature'])

    # Desired shape
    desired_shape = (1, 137)

    # Convert to dense array, pad, and convert back to sparse matrix if needed
    dense_tfidf_vector = tfidf_vector.toarray()
    if dense_tfidf_vector.shape[1] < desired_shape[1]:
        dense_tfidf_vector = np.pad(dense_tfidf_vector, ((0, 0), (0, desired_shape[1] - dense_tfidf_vector.shape[1])), mode='constant')
        dense_tfidf_vector = dense_tfidf_vector[:, :desired_shape[1]]
    else:
        dense_tfidf_vector = dense_tfidf_vector[:, :desired_shape[1]]

# You can convert the dense array back to sparse matrix if needed
# tfidf_vector = sparse.csr_matrix(dense_tfidf_vector)

    # Convert other features to numpy array
    other_features = df[['has_company_logo', 'has_questions', 'telecommuting']].values
    # Combine TF-IDF transformed textual features with other features
    X = np.concatenate((dense_tfidf_vector, other_features), axis=1)
    X = X.reshape(1,-1);
    #print(X.shape)
    output = model.predict(X);

    return output[0];  

csv_file_path = 'C:\\Users\\snpav\\OneDrive\\Desktop\\DIC final_phase3\\DIC final\\src\\fake_job_postings _P2.csv'

@app.route('/')
def home():
    return render_template('webpage.html')

@app.route('/predict', methods = ["POST"])
def predict():
    title = request.form.get('title');
    countries = request.form.get('location');
    education = request.form.get('education');
    experience = request.form.get('experience');
    description = request.form.get('description');
    logo = request.form.get('logo');
    requirements = request.form.get('requirements');
    employment_type = request.form.get('employment_type');
    has_questions = request.form.get('has_questions');
    benefits = request.form.get('benefits');
    telecommuting = request.form.get('telecommuting');
    
    """if 'csvfile' not in request.files:
        return 'No file uploaded', 400
    
    csv_file = request.files['csvfile']
    
    if csv_file and csv_file.filename.endswith('.csv'):
        csv_data = csv_file.read().decode('utf-8')  # Read CSV data as string
        df = pd.read_csv(csv_data)
        get_graph(df);
        
        return 'CSV file uploaded and processed successfully'
    else:
        return 'Please upload a CSV file', 400"""
    
    #return request.form.values()
    output = analyze(title, countries, education, experience, description, logo, requirements, employment_type, has_questions, benefits, telecommuting);
    if output==0:
        get_graph();
        return render_template('goodtogomsg.html')
        
    else:
        return render_template('cauctionmsg.html')


    
    
if __name__ == '__main__':
    app.run()