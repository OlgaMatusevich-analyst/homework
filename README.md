# homework
week 2 homework
Prompt for this: 

Hello! 
I'm a student in the "Neural Networks and Deep Learning" course, and I was given homework to create a web application and fix any errors that might appear. For this, I'm attaching the prompt you need to use to create the application, along with course presentations and a screenshot highlighting the errors you need to avoid. I'm also attaching the datasets that will be used to create the model and a link to my GitHub account (https://github.com/OlgaMatusevich-analyst). 

Complete the assignment using this script and the description in the presentation + CORRECT ALL ERRORS in the presentation. Prompt " Output two separate code files: first index.html (for HTML structure and UI), second app.js (for JavaScript logic). 

The app is a shallow binary classifier on Kaggle Titanic dataset using TensorFlow.js, running entirely in the browser (no server), ready for GitHub Pages. Use TensorFlow.js CDN 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest' and tfjs-vis CDN 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-vis@latest'. Link app.js from index.html. Follow this workflow: 

 1.**Layout in index.html**: Sections for Data Load, Preprocessing, Model, Training, Metrics (ROC-AUC slider), Prediction, Export. Add file inputs for train.csv/test.csv. Use basic CSS for responsiveness. 

2. **Data Schema in app.js**: Target: Survived (0/1). Features: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked. Identifier: PassengerId (exclude). // Reuse note: Swap schema for other datasets. 

3. **Load & Inspect in app.js**: Load CSV via fetch or file input (handle errors); show preview table, shape, missing %; chart survival by Sex/Pclass with tfjs-vis bars. 

4. **Preprocessing in app.js**: Impute Age (median), Embarked (mode); standardize Age/Fare; one-hot Sex/Pclass/Embarked. Toggle FamilySize=SibSp+Parch+1, IsAlone=(FamilySize==1). Print features/shapes. 

5. **Model in app.js**: tf.sequential: Dense(16, 'relu') (single hidden layer), Dense(1, 'sigmoid'). Compile 'adam', 'binaryCrossentropy', ['accuracy']. Print summary. 

6. **Training in app.js**: 80/20 stratified split; train 50 epochs, batch 32; tfjs-vis fitCallbacks for live loss/accuracy plots; early stopping on val_loss (patience=5). 

7. **Metrics in app.js**: Compute ROC/AUC from val probs; plot ROC; slider (0-1) updates confusion matrix, Precision/Recall/F1 dynamically. 

8. **Inference & Export in app.js**: Predict test.csv probs; apply threshold for Survived; download submission.csv (PassengerId, Survived), probabilities.csv; model.save('downloads://titanic-tfjs'). 

9. **Deployment Notes in index.html**: Add text: 'Create public GitHub repo, commit index.html/app.js, enable Pages (main/root), test URL.' Include English comments in code. Make interactive with buttons (Train, Evaluate, Predict). Handle errors (e.g., alerts for missing files/invalid data). Ensure reusable by commenting schema swap points. "
