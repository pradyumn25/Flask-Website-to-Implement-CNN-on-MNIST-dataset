from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPool2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/plot",methods=["POST","GET"])
def plot():
    if request.method == "POST":
        num = request.form.get("num")
        num = int(num)
        (x_train,y_train),(x_test,y_test) = mnist.load_data()
        y_cat_test = to_categorical(y_test)
        y_cat_train = to_categorical(y_train)
        x_train = x_train/255
        x_test = x_test/255
        x_train = x_train.reshape(60000,28,28,1)
        x_test = x_test.reshape(10000,28,28,1)
        model = Sequential()

        model.add(Conv2D(filters=32,kernel_size=(4,4),input_shape=(28,28,1),activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(128,activation='relu'))

        #output layer
        model.add(Dense(10,activation='softmax'))#softmax bcz of multi class classification
        model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
        early_stop = EarlyStopping(monitor='val_loss',patience=1)
        model.fit(x_train,y_cat_train,epochs=10,validation_data=(x_test,y_cat_test),callbacks=[early_stop])
        metrics = pd.DataFrame(model.history.history)
        pred = np.argmax(model.predict(x_test), axis=-1)

        img = BytesIO()

        pred1 = np.where(pred == num)
        index = pred1[0][0]
        my_number = x_test[index]
        plt.imshow(my_number.reshape(28,28))

        acc = accuracy_score(y_test,pred)
        acc = acc*100
        acc = str(acc)

        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("plot.html",plot_url=plot_url,acc = acc)

app.run(debug=True)