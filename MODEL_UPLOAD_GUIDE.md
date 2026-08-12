##Upload Large .h5 Model to GitHub using Git LFS

Use the following steps to upload your large cat_dog_cnn_model.h5 file to GitHub with Git LFS (Large File Storage).

1. Install and initialize Git LFS
- git lfs install

---

2. Clone your GitHub repository
- git clone https://github.com/YOUR_USERNAME/CNN-Cat-and-Dog-Classification.git

---

3. Open the repository folder
- cd CNN-Cat-and-Dog-Classification

---

4. Configure your GitHub username
- git config --global user.name "USERNAME"

---

5. Configure the email associated with your GitHub account
- git config --global user.email "YOUR_GITHUB_EMAIL"

6. Tell Git LFS to track the large H5 model file
- git lfs track "*.h5"

7. Add the LFS configuration and model file
- git add .gitattributes cat_dog_cnn_model.h5

8. Commit the large model file
- git commit -m "Add h5 file via Git LFS"

9. Push the commit to the main branch
- git push origin main
