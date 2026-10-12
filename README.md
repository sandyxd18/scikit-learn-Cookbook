# scikit-learn Cookbook (Python Implementation)

Repositori ini berisi implementasi praktis konsep *machine learning* menggunakan Python dan ekosistem scikit-learn. Materi dan kode pada repositori ini mengacu pada buku **"scikit-learn Cookbook" (Third Edition)** oleh John Sukup.

Setiap bab disusun dalam format Jupyter Notebook interaktif yang menggabungkan penjelasan konsep teoritis, arsitektur API scikit-learn, serta implementasi kode langsung lengkap dengan visualisasi dan hasil eksekusi model.

---

## Daftar Isi dan Rincian Tiap Bab

### [Bab 1: Common Conventions and API Elements of scikit-learn](Bab1_Common_Conventions_and_API_Elements.ipynb)
Membahas fondasi arsitektur dan pola desain API scikit-learn yang menjadi standar di seluruh buku:
- **Filosofi Desain scikit-learn**: Konsistensi antarmuka (*consistent interface*), inspeksi atribut terestimasi melalui konvensi *trailing underscore* (`_`), modularitas komponen, dan kompatibilitas dengan struktur data NumPy/Pandas.
- **Estimator dan Method `fit()`**: Objek pemodelan dasar untuk pembelajaran statistik, mekanisme pelatihan model via `.fit()`, serta ragam antarmuka turunannya (Classifier, Regressor, Clusterer).
- **Transformer dan Method `transform()`**: Komponen pra-pemrosesan data dengan method `.fit()`, `.transform()`, dan `.fit_transform()` untuk transformasi fitur tanpa mengubah jumlah sampel.
- **Custom Estimator dan Transformer**: Pembuatan komponen kustom yang kompatibel penuh dengan ekosistem scikit-learn menggunakan kelas dasar `BaseEstimator` dan `TransformerMixin`.
- **Pipeline dan Otomatisasi Alur Kerja**: Penggabungan rantai transformasi dan estimator akhir dalam satu objek `Pipeline` untuk mencegah kebocoran data (*data leakage*) dan mempermudah deployment.
- **Atribut dan Method Inti**: Penggunaan method prediksi (`.predict()`, `.predict_proba()`), evaluasi performa (`.score()`), serta inspeksi hyperparameter via `.get_params()` dan `.set_params()`.
- **Hyperparameter Tuning**: Pencarian kombinasi parameter optimal secara sistematis menggunakan `GridSearchCV` (eksplorasi menyeluruh) dan `RandomizedSearchCV` (sampling distribusi probabilistik).
- **Metadata dan Tags**: Inspeksi kapabilitas dan karakteristik estimator melalui `__sklearn_tags__()` serta mekanisme perutean metadata tambahan.
- **Best Practices Penggunaan API**: Panduan penulisan kode standar produksi, penerapan validasi silang (*cross-validation*), dan pencegahan kesalahan metodologis umum.

### [Bab 2: Pre-Model Workflow and Data Preprocessing](Bab2_Pre_Model_Workflow_and_Data_Preprocessing.ipynb)
Membahas teknik penyiapan dan transformasi data mentah sebelum tahap pemodelan prediktif:
- **Kualitas Data Mentah dan Risiko Pemodelan**: Dampak nilai hilang (*missing values*), pencilan (*outliers*), variabel kategorikal, disparitas skala antar-fitur, dan bahaya *data leakage*.
- **Penanganan Missing Data (Imputasi)**:
  - `SimpleImputer()`: Imputasi univariat berbasis statistik deskriptif (mean, median, most frequent, atau nilai konstan).
  - `KNNImputer()`: Imputasi berbasis jarak kedekatan k-tetangga terdekat antar sampel data.
  - `IterativeImputer()` (MICE): Pendekatan multivariat berbasis chained equations yang memodelkan tiap fitur hilang sebagai fungsi regresi dari fitur lainnya.
- **Teknik Penskalaan Fitur (Scaling)**:
  - `StandardScaler()`: Standarisasi fitur ke distribusi dengan rata-rata 0 dan standar deviasi 1 (z-score).
  - `MinMaxScaler()`: Transformasi skala fitur ke rentang batas tertentu (umumnya 0 sampai 1).
  - `Normalizer()`: Penskalaan ulang tiap baris sampel secara independen agar memiliki norma satuan (L1, L2, atau Max norm).
- **Encoding Variabel Kategorikal**:
  - `OneHotEncoder()`: Transformasi variabel kategori nominal ke matriks biner (*dummy variables*).
  - `LabelEncoder()`: Konversi label target kategorikal menjadi indeks integer.
  - `ColumnTransformer()`: Penerapan transformasi yang berbeda secara terpisah antara kolom numerik dan kategorikal dalam satu alur pipeline.
- **Feature Engineering dan Seleksi Fitur**:
  - `PolynomialFeatures()`: Pembangkitan fitur polinomial dan interaksi non-linear antar-variabel.
  - `KBinsDiscretizer()`: Diskretisasi fitur kontinu ke dalam *bins* diskrit (uniform, quantile, kmeans).
  - Seleksi Fitur: Metode eliminasi rekursif dengan `RFE()` dan seleksi berbasis koefisien/bobot model dengan `SelectFromModel()`.
- **Latihan Komprehensif (California Housing Dataset)**: Alur kerja terintegrasi preprocessing, pembagian train-test, pelatihan model regresi (`RandomForestRegressor` dan `LinearRegression`), serta evaluasi metrik RMSE dan $R^2$.

### [Bab 3: Dimensionality Reduction Techniques](Bab3_Dimensionality_Reduction_Techniques.ipynb)
Membahas teknik kompresi dan reduksi dimensi fitur untuk efisiensi komputasi, mitigasi overfitting, dan visualisasi:
- **Urgensi Reduksi Dimensi**: Mengatasi fenomena *curse of dimensionality*, mempercepat waktu training, membuang *noise*, dan memproyeksikan data ke bidang 2D/3D.
- **Principal Component Analysis (PCA)**:
  - Metode reduksi dimensi *unsupervised* linear berbasis dekomposisi matriks kovarians.
  - Analisis rasio variansi terjelaskan (*explained variance ratio*) untuk menentukan jumlah komponen optimal.
- **Linear Discriminant Analysis (LDA)**:
  - Metode reduksi dimensi *supervised* yang memaksimalkan jarak pemisahan antar-kelas sambil meminimalkan sebaran variansi di dalam kelas (*class separability*).
  - Perbandingan perbedaan mendasar tujuan dan asumsi matematis antara PCA dan LDA.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**:
  - Metode reduksi dimensi non-linear probabilistik berbasis divergensi Kullback-Leibler untuk mempertahankan struktur lokal dalam visualisasi manifold berdimensi tinggi.
- **Panduan Pemilihan Teknik Reduksi Dimensi**: Kriteria pemilihan metode berdasarkan ketersediaan label target, tujuan analitik (ekstraksi fitur vs. eksplorasi visual), efisiensi komputasi, dan interpretabilitas.
- **Latihan Praktis**: Integrasi PCA dengan `LogisticRegression` pada dataset Wine/Digits serta penerapan t-SNE yang dikombinasikan dengan pengelompokan `KMeans`.

### [Bab 4: Building Models with Distance Metrics and Nearest Neighbors](Bab4_Distance_Metrics_and_Nearest_Neighbors.ipynb)
Membahas pemodelan berbasis metrik jarak, khususnya algoritma K-Nearest Neighbors (KNN):
- **Pengantar Metrik Jarak**:
  - *Euclidean Distance* ($L_2$ norm): Jarak garis lurus standar dalam ruang berdimensi n.
  - *Manhattan Distance* ($L_1$ norm / city block): Penjumlahan selisih absolut koordinat, lebih tangguh terhadap outlier pada dimensi tinggi.
  - *Minkowski Distance*: Bentuk umum parametrik dengan parameter $p$ ($p=1$ setara Manhattan, $p=2$ setara Euclidean).
- **Mekanisme Algoritma KNN**:
  - Prinsip *lazy learning* / *instance-based learning* tanpa pembentukan fungsi eksplisit selama training.
  - Penentuan nilai hyperparameter $k$ (keseimbangan bias vs. variansi).
  - Mekanisme voting mayoritas untuk klasifikasi serta agregasi rata-rata (uniform vs. distance-weighted) untuk regresi.
- **Hyperparameter Tuning pada KNN**: Optimasi nilai $k$, pilihan metrik jarak (`metric`), dan skema bobot tetangga (`weights`) menggunakan `GridSearchCV`.
- **Evaluasi Performa Komprehensif**:
  - Metrik evaluasi klasifikasi: Confusion Matrix, Precision, Recall, F1-Score, Accuracy, Macro Average, dan Weighted Average.
  - Analisis kurva pembelajaran (*learning curves*) dan kestabilan skor via `cross_val_score`.
- **Latihan Praktis**: Pemodelan klasifikasi batas keputusan non-linear pada data sintetis (`make_circles`) serta dataset multi-kelas riil (Wine dan Iris).

### [Bab 5: Linear Models and Regularization](Bab5_Linear_Models_and_Regularization.ipynb)
Membahas model linear untuk estimasi regresi serta teknik regularisasi untuk mencegah overfitting:
- **Pengantar Model Linear**: Konsep regresi linear Ordinary Least Squares (OLS), estimasi slope dan intercept, serta fungsi objektif residual sum of squares (RSS).
- **Teknik Regularisasi**:
  - *Ridge Regression* (Regularisasi $L_2$): Penambahan penalti kuadrat magnitude koefisien ($\lambda \sum w_j^2$), menyusutkan koefisien mendekati nol untuk menangani multikolinearitas.
  - *Lasso Regression* (Regularisasi $L_1$): Penambahan penalti nilai absolut magnitude koefisien ($\lambda \sum |w_j|$), menghasilkan model *sparse* dengan membuat koefisien fitur yang tidak informatif menjadi tepat nol (seleksi fitur otomatis).
  - *ElasticNet Regression*: Kombinasi penalti $L_1$ dan $L_2$ melalui parameter `l1_ratio`, memberikan fleksibilitas ekstra pada data dengan fitur yang saling berkorelasi tinggi.
- **Teori dan Dinamika Jalur Koefisien (Regularization Paths)**: Evaluasi penyusutan nilai koefisien terhadap peningkatan kekuatan penalti $\alpha$ dalam skala logaritmik.
- **Penanganan Hubungan Non-Linear**:
  - *Polynomial Regression*: Transformasi fitur non-linear menggunakan `PolynomialFeatures` yang dipadukan dengan model linear terregularisasi.
  - *Spline Interpolation*: Pemanfaatan `SplineTransformer` (B-spline basis) untuk menangani kelengkungan fungsi secara piecewise tanpa osilasi ekstrem di ujung rentang data.
- **Latihan Praktis Komparatif**: Perbandingan langsung performa OLS vs. Ridge vs. Lasso vs. ElasticNet pada data sintetis (`make_regression`), lengkap dengan visualisasi MSE, $R^2$, dan plot koefisien.

### [Bab 6: Advanced Logistic Regression and Extensions](Bab6_Advanced_Logistic_Regression_and_Extensions.ipynb)
Membahas teknik lanjutan regresi logistik untuk klasifikasi multikelas, penalti regularisasi, dan multilabel:
- **Overview of Logistic Regression**: Prinsip dasar fungsi sigmoid, transformasi odds ke log-odds (logit), cross-entropy loss, dan estimasi parameter probabilitas untuk klasifikasi biner.
- **Multiclass Classification Techniques**:
  - *One-vs-Rest (OvR)*: Pelatihan satu pengklasifikasi biner per kelas independen.
  - *Multinomial (Softmax Regression)*: Normalisasi probabilitas bersama seluruh kelas via fungsi softmax dan optimasi multinomial loss.
- **Regularization in Logistic Regression**:
  - Perbandingan penalti $L_1$ (sparsity/seleksi fitur otomatis), $L_2$ (penyusutan koefisien), dan ElasticNet.
  - Efek parameter invers kekuatan regularisasi $C$ dan pemilihan solver (`lbfgs`, `saga`, `liblinear`).
  - Visualisasi dampak intensitas regularisasi terhadap besaran koefisien model.
- **Multilabel Classification Concepts**:
  - Penanganan masalah di mana sampel dapat memiliki lebih dari satu label secara simultan.
  - Implementasi pendekatan Binary Relevance dan pembungkus `MultiOutputClassifier`.
  - Visualisasi matriks co-occurrence antar-label.
- **Model Evaluation Metrics & Visualisasi**: Evaluasi log loss, confusion matrix, ROC-AUC multikelas, dan visualisasi pemetaan batas keputusan 2D berbasis meshgrid probabilitas.

### [Bab 7: Support Vector Machines and Kernel Methods](Bab7_Support_Vector_Machines_and_Kernel_Methods.ipynb)
Membahas algoritma Support Vector Machine (SVM) dan metode kernel untuk batas keputusan linear dan non-linear:
- **Introduction to SVMs**: Konsep *maximum margin hyperplane*, identifikasi support vectors, pemodelan klasifikasi (`SVC`) dan regresi (`SVR` dengan parameter $\epsilon$-insensitive tube).
- **Kernel Functions and Their Applications**:
  - Konsep Kernel Trick: Memetakan ruang input berdimensi rendah ke ruang fitur berdimensi tinggi secara implisit tanpa komputasi eksplisit.
  - Perbandingan performa kernel Linear, Polynomial (`poly`), dan Radial Basis Function (`rbf`) pada data terpisah non-linear (`make_moons`).
- **Tuning SVM Parameters**: Eksplorasi peran parameter $C$ (kompromi antara ukuran margin dan toleransi kesalahan klasifikasi) serta parameter $\gamma$ (lebar jangkauan pengaruh tiap support vector pada kernel RBF).
- **SVMs in High-Dimensional Spaces**: Karakteristik dan ketahanan SVM terhadap kutukan dimensi (*curse of dimensionality*) pada data dengan jumlah fitur sangat banyak ($p \gg n$), disertai visualisasi proyeksi komponen PCA.
- **Evaluating SVM Models & Latihan Praktis**: Integrasi pipeline `StandardScaler` dan `SVC`, optimasi hyperparameter dengan `GridSearchCV`, serta visualisasi kontur batas keputusan dan margin pendukung.

### [Bab 8: Tree-Based Algorithms and Ensemble Methods](Bab8_Tree_Based_Algorithms_and_Ensemble_Methods.ipynb)
Membahas model berbasis pohon keputusan dan teknik ensembel (Bagging, Random Forest, dan Boosting):
- **Introduction to Decision Trees**: Algoritma partisi rekursif (`DecisionTreeClassifier`), kriteria pemisahan homogenitas (Gini Impurity vs. Entropy), dan visualisasi struktur pohon via `plot_tree`.
- **Mengendalikan Kompleksitas Tree**: Mitigasi risiko overfitting melalui pembatasan kedalaman (`max_depth`), sampel minimum per daun (`min_samples_leaf`), serta teknik pemangkasan Cost-Complexity Pruning (`ccp_alpha`).
- **Random Forests and Bagging**:
  - *Bagging* (Bootstrap Aggregating): Mereduksi variansi model dasar melalui agregasi hasil voting dari banyak model yang dilatih pada subset sampel bootstrap (`BaggingClassifier`).
  - *Random Forest* (`RandomForestClassifier`): Ensembel pohon keputusan yang menggabungkan sampling bootstrap data dengan pengacakan subset fitur (*feature subsampling*) pada setiap pemisahan simpul.
- **Gradient Boosting Machines**:
  - Konsep pembelajaran boosting sekuensial via `GradientBoostingClassifier` di mana tiap pohon baru memprediksi pseudo-residual kesalahan dari pohon-pohon sebelumnya.
  - Penyetelan parameter laju pembelajaran (*learning rate*) dan jumlah pohon (`n_estimators`).
- **Hyperparameter Tuning & Perbandingan Ensemble**: Analisis komparatif kinerja Decision Tree vs. Random Forest vs. Gradient Boosting menggunakan cross-validation, serta inspeksi kontribusi prediktor melalui *feature importances*.

---

## Dataset yang Digunakan

Implementasi kode pada notebook memanfaatkan dataset bawaan (*built-in*) scikit-learn, data sintetis, maupun data riil eksternal:

- **California Housing Dataset**: Data sensus perumahan California untuk simulasi preprocessing lengkap dan pemodelan regresi (Bab 2).
- **Iris Dataset**: Dataset morfologi bunga Iris (`load_iris`) untuk demonstrasi klasifikasi multi-kelas, reduksi dimensi, dan evaluasi metrik jarak KNN (Bab 3, 4, 6, 7, 8).
- **Wine Dataset**: Data profil kimiawi tiga kultivar anggur (`load_wine`) untuk evaluasi pemisahan kelas PCA, LDA, dan tuning model KNN (Bab 3, 4).
- **Optical Recognition of Handwritten Digits**: Dataset citra digit angka 8x8 piksel (`load_digits`) untuk kompresi dimensi, ekstraksi fitur, dan visualisasi t-SNE (Bab 3).
- **Breast Cancer Wisconsin Dataset**: Dataset diagnosis tumor payudara (`load_breast_cancer`) untuk klasifikasi biner, tuning regularisasi $C$, dan evaluasi SVM pada data dimensi tinggi (Bab 6, 7).
- **Synthetic Datasets (`make_circles`, `make_regression`, `make_moons`, `make_multilabel_classification`)**: Dataset buatan untuk visualisasi batas keputusan non-linear, evaluasi regularisasi linear, pengujian kernel SVM, dan demonstrasi klasifikasi multilabel (Bab 4, 5, 6, 7).

---

## Teknologi dan Library Python

Proyek ini dibangun menggunakan pustaka ekosistem data science dan machine learning standar industri:

- **Python 3**: Bahasa pemrograman utama.
- **scikit-learn**: Pustaka utama untuk estimator, transformer, metrik evaluasi, seleksi fitur, algoritma reduksi dimensi, dan pipeline pemodelan.
- **pandas**: Manipulasi, transformasi, dan pembersihan data tabular.
- **numpy**: Komputasi array numerik multidimensi dan operasi aljabar linear.
- **matplotlib** & **seaborn**: Visualisasi batas keputusan, scatterplot komponen utama (PCA/t-SNE), kurva evaluasi model, dan grafik jalur koefisien regularisasi.
- **Jupyter Notebook**: Antarmuka interaktif eksekusi kode dan penyajian visualisasi.

---

## Cara Menjalankan Proyek

1. **Clone repositori ini**:
   ```bash
   git clone https://github.com/sandyxd18/scikit-learn-Cookbook.git
   cd scikit-learn-Cookbook
   ```

2. **Siapkan virtual environment (opsional tetapi disarankan)**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Untuk Linux/macOS
   # venv\Scripts\activate   # Untuk Windows
   ```

3. **Install dependensi yang diperlukan**:
   ```bash
   pip install scikit-learn pandas numpy matplotlib seaborn jupyter
   ```

4. **Jalankan Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Buka berkas notebook bab yang ingin dipelajari (misalnya `Bab1_Common_Conventions_and_API_Elements.ipynb` hingga `Bab8_Tree_Based_Algorithms_and_Ensemble_Methods.ipynb`). Seluruh sel kode di dalam notebook sudah tersimpan bersama luaran eksekusinya (*pre-computed outputs*).
