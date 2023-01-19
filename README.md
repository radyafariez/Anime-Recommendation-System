# Anime Recommendation System - Mohammad Radya Fariez

----------------------------------------------------
Disusun oleh : Mohammad Radya Fariez

Berikut merupakan laporan submission project terakhir pada kelas Dicoding (Machine Learning Terapan/Applied Machine Learning)

----------------------------------------------------

### Project Overview

Proyek kali ini akan membahas mengenai Recommendation System untuk Anime. Dapat dikatakan sebuah perusahaan di bidang hiburan/_entertainment_ sedang berfokus untuk mengembangkan web mereka dengan mengembangkan beberapa fitur, salah satunya Recommender System.

Recommender System itu sendiri (dalam hal ini) merupakan hasil dari pengembangan Machine Learning yang bertujuan untuk memudahkan pengguna aplikasi atau web dalam memilih konten dengan cara menampilkan konten tersebut berdasarkan apa yang yang telah dilihat sebelumnya (Content-Based Filtering). 

**Latar Belakang**

Industri film/sinematik di era saat ini menyajikan jenis tontonan yang beragam, dimulai dari jenis penayangan yang dapat dinikmati di TV, Bioskop, bahkan digital product seperti Netflix, Disney+ dan sebagainya. Begitu pula dengan ragam jenis tontonan, saat ini jenis tontonan tidak hanya sebatas _real action_ dan animasi saja, melainkan anime. 

Mengutip dari laman [Wikipedia](https://id.wikipedia.org/wiki/Anime), Anime merupakan animasi asal Jepang yang digambar dengan tangan maupun teknologi komputer, sehingga tak dapat dipungkiri Anime memiliki ciri khas tersendiri. Selain itu, alur cerita yang unik dan beragam serta _relatable_ dengan kehidupan saat ini menjadi alasan bagi para penggemar di seluruh dunia, termasuk Indonesia.

Adapun, tayangan Anime saat ini diwarnai dengan beragamn genre, contohnya genre komedi, horror dan petualangan, sehingga penonton dapat memiliki opsi untuk memilih tontonan yang disuka. Oleh karena itu, hal tersebut bisa dijadikan peluang untuk meningkatkan kualitas web dan aplikasi dengan menciptakan fitur Recommender System dengan tujuan meningkatkan jumlah pengunjung/_traffic_. Dengan adanya fitur tersebut, pengguna secara langsung akan merasakan kemudahan dalam memilih dan melakukan seleksi terhadap konten yang akan dinikmati.

----------------------------------------------------

### Business Understanding

Berdasarkan ulasan mengenai Project Overview dan latar belakang, maka dapat diuraikan suatu problem statements dan tujuan sebagai berikut.

**Problem Statements**

Problem statement pada proyek kali ini, diantaranya :

1. Bagaimana cara menentukan suatu Recommender System dengan hasil yang baik dan optimal?

2. Metode Recommendation System apa yang paling cocok dengan permasalahan pada proyek ini?

**Goals**

Tujuan yang dapat diraih berdasarkan problem statement diatas, yaitu :

1. Melakukan evaluasi terhadap hasil performa model Machine Learning

2. Metode yang paling sesuai dengan permasalahan ini, yaitu dengan menerapkan metode _Content-Based Filtering_.

----------------------------------------------------

### Data Understanding

Dataset yang digunakan pada proyek ini bersumber dari Kaggle yang dapat diunduh di [Anime Dataset](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database). Dari dataset tersebut, didapatkan informasi sebagai berikut :

- Dataset merupakan data yang memiliki format CSV
- Terdapat 2 _file_ CSV pada dataset, diantaranya 'anime.csv' dan 'rating.csv'. Deskripsi kedua _file_ tersebut masing - masing dapat dideskripsikan di bawah ini.

--------------------------

**Deskripsi Dataset**

**Dataset 'anime.csv'**

Pada dataset 'anime.csv', dapat dideskripsikan beberapa hal, diantaranya :

- Memiliki 1.156 sampel data
- Di dalam dataset terdapat 7 fitur
- Dari 7 fitur tersebut, terdapat 4 fitur numerik dan 3 fitur kategorik.

Beberapa variable fitur pada dataset dapat didefinisikan sebagai berikut :

- 	**anime_id** : ID atau identitas konten anime
- 	**name**     : Nama konten anime
- 	**genre**    : Genre pada anime 
- 	**type**     : Jenis platform yang ditayangkan
- 	**episodes** : Jumlah episod
- 	**rating**   : Rate atau nilai konten per user
- 	**members**  : Jumlah member.

**Dataset 'rating.csv'**

Pada dataset 'rating.csv' berisi nilai masing - masing konten anime dari beberapa user yang memiliki karakter berikut :

- Memiliki sejumlah 7.813.737 sampel data
- Terdapat 3 fitur pada data
- Pada data 'rating.csv', memiliki 3 fitur yang ketiganya merupakan fitur numerik.

Variable fitur pada dataset masing - masing dapat didefinisikan sebagai berikut :

- 	**anime_id** : ID atau identitas konten anime
- 	**rating**   : Rate atau nilai konten per user
- 	**user_id**  : Identitas user.

----------------------------------------------------

### Exploratory Data Analysis

**Univariate Data Analysis**

Masing - masing fitur dapat direpresentasikan dalam bentuk grafik berikut :

**- Fitur Genre**

![genre](https://user-images.githubusercontent.com/109395960/213077027-ff09b722-48d8-4fbb-bad3-08aa8d9f29de.png)

Pada grafik fitur genre berdasarkan gambar di atas, dapat disimpulkan bahwa terdapat 12 genre dengan kesimpulan genre Comedy berada di urutan teratas sebagai genre paling banyak di dataset, sedangkan genre Supernatural berada di urutan paling bawah. Rincian persentase grafik tersebut dapat dilihat pada tabel dibawah ini.

|               |jumlah Anime | Persentase (%) |
|---------------|-------------|----------------|
|Comedy         |         523 |       45.2     |
|Fantasy        |         114 |        9.9     |
|Drama          |         107 |        9.3     |
|Slice of Life  |          99 |        8.6     |
|Adventure      |          79 |        6.8     |
|Historical     |          68 |        5.9     |
|Action         |          53 |        4.6     |
|Sports         |          44 |        3.8     |
|Horror         |          21 |        1.8     |
|Mystery        |          19 |        1.6     |
|Romance        |          15 |        1.3     |
|Supernatural   |          14 |        1.2     |

Dapat diketahui bahwa, genre Comedy memiliki porsi sebanyak 45.2%, diikuti genre Fantasy dan Drama dengan masing - masing 9.9% dan 9.3%. Sedangkan genre Supernatural yang memiliki porsi paling sedikit (1.2%).

**- Fitur Rating**

Kemudian pada fitur rating, dapat diperoleh suatu grafik representasi sebagai berikut :

![ratings](https://user-images.githubusercontent.com/109395960/213083816-b342537b-8d53-438b-a9d3-f4ba081ff6cc.png)

Dari grafik fitur rating, dapat ditarik beberapa kesimpulan, diantaranya :

- Rating atau penilaian konten Anime secara keseluruhan bernilai minimum -1 dan maksimum 10
- Rating paling banyak bernilai masing - masing secara berurutan 8, 7 dan 9. Artinya, user paling banyak memberi penilaian di angka 8, disusul angka 7 dan 9.

----------------------------------------------------

### Data Preparation

Dilakukan beberapa step pada proses Data Preparation, diantaranya :

**- Menggabungkan Dataset**

Dalam proses ini, dataset 'anime.csv' dan 'ratings.csv' digabungkan untuk memperoleh jumlah aktual dari beberapa fitur. Dari langkah ini, dapat diketahui bahwa :

- Jumlah aktual fitur **anime_id** berjumlah 11363 data

- Jumlah rating aktual secara keseluruhan sebanyak 380 data.

**- Mengatasi Missing Value**

Langkah selanjutnya adalah mendapatkan jumlah _missing value_ dengan menggunakan command __isnull().sum()_ dan menghapus _missing value_ (NaN) untuk memperoleh hasil model yang optimal.

**- Sorting Data**

Mengurutkan data dari yang terkecil berdasarkan fitur anime_id

**- Menghapus Data yang Terduplikasi**

Dataset yang kita peroleh terkadang berisi data yang terduplikasi atau kembar. Untuk mengatsi hal tersebut, dilakukan drop pada data yang terduplikasi

**- Konversi Data menjadi Bentuk List**

Untuk mempermudah proses pengolahan data, dilakukan konversi data series menjadi format _list_.

Dengan melakukan langkah - langkah Data Preparation, didapatkan sampel data yang dapat diolah menjadi sebanyak 1143 data.

----------------------------------------------------

### Modelling

Di proyek Recommender System ini telah selesai tahap Data Preparation, tahap selanjutnya memasuki proses _modelling_ dengan beberapa langkah berikut :

**- Inisialisasi dan Vektorisasi TF-IDF**

Pada tahap ini dilakukan vektorisasi dengan fungsi _TfidfVectorizer()_ dan melakukan mapping array dari fitur index integer ke fitur nama untuk mengidentifikasi korelasi judul konten anime dengan genre dan fitment dalam bentuk matriks. Pada tahap ini, output yang dihasilkan berupa _list_ dari sejumlah genre pada konten dan korelasi berupa matriks.

**- Cosine Similarity**

Dilakukan pengukuran tingkat kesamaan antara kedua vektor pada matriks berdasarkan hasil pada tahap sebelumnya. 

**- Membuat Fungsi Anime Recommendation**

Melakukan konversi dataframe ke format NumPy dan mendefinisikan fungsi _animes_recommendation_ dengan melakukan pengambilan data menggunakan _agpartition_  untuk melakukan partisi secara tidak langsung. Lalu dilakukan pengambilan data dengan angka similarity terbesar dan melakukan drop pada titel konten agar titel tersebut tidak muncul di dalam daftar rekomendasi.

### Result

Setelah melakukan modelling, dilakukan pengujian pada salah satu sampel konten Anime secara acak. Dalam hal ini, dilakukan pengujian terhadap konten yang berjudul 'Wolf Daddy' dan menghasilkan output berikut :

|      |id      | titles     | genre   |
|------|--------|------------|---------|
|118   |  4099  | Wolf Daddy | Fantasy |

Setelah itu, dilakukan pengujian terhadap rekomendasi konten Anime lainnya yang memilki kesamaan genre dengan memanggil fungsi yang telah didefinisikan (animes_recommendation). Maka, didapatkan output berikut :


|   |titles                                             |genre  |
|---|---------------------------------------------------|-------|
|0	|Shiawasette Naani	                                |Fantasy|
|1	|Escha &amp; Logy no Atelier: Tasogare no Sora ...	|Fantasy|
|2	|Lance N&#039; Masques	                            |Fantasy|
|3	|Fushigi na Elevator	                              |Fantasy|
|4	|Spectral Force Chronicle Divergence	              |Fantasy|
|5	|Santa Company	                                    |Fantasy|
|6	|Kudan	                                            |Fantasy|
|7	|Hana to Shounen	                                  |Fantasy|
|8	|Ukkari Pénélope OVA	                              |Fantasy|
|9	|on-chan, Yume Power Daibouken!                   	|Fantasy|

Berdasarkan output yang diperoleh, dapat diketahui bahwa model mampu memberikan rekomendasi dengan korelasi berdasarkan genre.

----------------------------------------------------

### Evaluation

Evaluasi yang dilakukan sesuai dengan kategori model pada proyek ini, yaitu metrik _Precision_ atau ketentuan presisi. Maka, evaluasi dapat dilakukan berdasarkan formula berikut :

Recommender System Precision :

$$
P = {a \over b}
$$


dengan keterangan bahwa, 

a = number of our relevant recommendations/jumlah rekomendasi konten yang relevan,

b = number of items we recommended/jumlah item atau konten yang kita rekomendasikan.

Dengan formula tersebut, kita dapat menghitung hasil dari model yang telah didapat sebelumnya, yaitu :

$$
P = {10 \over 10} = 1
$$

Maka, jika dikonversi menjadi nilai persen akan menjadi 100%.

