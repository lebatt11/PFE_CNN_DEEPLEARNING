#import secrets

#secret_key = secrets.token_hex(16)
#print(secret_key)


SECRET_KEY = '747bc7dcf0e0f42ca1d660626e167635'



import io
import cv2
import base64
import numpy as np
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import load_model # type: ignore
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, request, redirect, url_for, flash, send_file
from werkzeug.security import check_password_hash, generate_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user


# chargement du modéle :
Model_Path= 'models/system.h5'
model = load_model(Model_Path)



# définition et configuration de l'application :
app = Flask(__name__)
app.config['SECRET_KEY'] = '747bc7dcf0e0f42ca1d660626e167635'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///main.db'
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

# Create the database and add the user
with app.app_context():
    db.create_all()
    if not User.query.filter_by(username='expert').first():
        hashed_password = generate_password_hash('django', method='pbkdf2:sha256')
        new_user = User(username='expert', password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('house'))
        else:
            flash('Invalid username or password')

    return render_template('login.html')


@app.route('/house')
@login_required
def house():
    return render_template('house.html')


@app.route('/index')
@login_required
def index():
    return render_template('index.html')

@app.route('/logout', methods =["POST"])
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))






@app.route('/',methods=['POST','GET'])
def predict():
    if request.method=='POST':
        imagefile= request.files["imagefile"]
        image_path ='./static/test' + imagefile.filename
        imagefile.save(image_path)
        
        
        # Lire l'image en utilisant OpenCV
        image = cv2.imread(image_path)
            
        # Redimensionner l'image à la taille cible
        img = cv2.resize(image, (150, 150))
            
        # Normaliser les valeurs de pixel pour être comprises entre 0 et 1
        img = img.astype('float32') / 255.0

        # Ajouter une dimension pour correspondre à l'entrée du modèle
        x = np.expand_dims(img, axis=0)

        # Prédire en utilisant le modèle
        prediction = model.predict(x)
        
        result = prediction[0][0]
        
        # seuil de confiance :
        threshold = 0.5
        
        if result > threshold :
            result='Ce patient est atteint de la pneumonie : résultat positif.'
        else:
            result='Ce patient est normal  : résultat négatif.'
            
        
        return render_template('result.html', prediction=result, result=result, imagePath=image_path)

    return render_template('home.html')



# Données réelles
training_data = {
    'accuracy': [0.9549, 0.9580, 0.9574, 0.9597, 0.9590, 0.9565, 0.9555, 0.9601, 0.9588, 0.9597],
    'loss': [0.1266, 0.1208, 0.1146, 0.1205, 0.1165, 0.1166, 0.1233, 0.1143, 0.1154, 0.1161]
}
validation_data = {
    'accuracy': [0.9145, 0.9145, 0.9145, 0.9161, 0.9128, 0.9178, 0.9128, 0.9161, 0.9145, 0.9161],
    'loss': [0.2464, 0.2446, 0.2452, 0.2450, 0.2486, 0.2232, 0.2458, 0.2361, 0.2415, 0.2396]
}
test_results = {
    'accuracy':0.9294,
    'precision': 0.9282,
    'recall': 0.9615,
    'f1_score': 0.9445
}
confusion_matrix = [[205, 29], [15, 375]]  # Example confusion matrix

@app.route('/performance')
def performance():
    img = io.BytesIO()
    plt.figure(figsize=(8, 3))

    # Training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(training_data['accuracy'], label='Training Accuracy')
    plt.plot(validation_data['accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(training_data['loss'], label='Training Loss')
    plt.plot(validation_data['loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Confusion matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    img_cm = io.BytesIO()
    plt.savefig(img_cm, format='png')
    img_cm.seek(0)
    confusion_matrix_url = base64.b64encode(img_cm.getvalue()).decode()

    return render_template('performance.html', plot_url=plot_url, confusion_matrix_url=confusion_matrix_url, test_results=test_results)

    

if __name__ == '__main__':
    app.run(debug=True)

