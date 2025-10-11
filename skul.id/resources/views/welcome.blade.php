<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.5/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            font-family: "Poppins", sans-serif;
            color: #252f35;
            font-size: 13px;
            background-color: #fff;
            margin: 0;
            padding: 0;
            min-height: 100vh;
            overflow-x: hidden;
        }

        header {
            display: flex;
            align-items: center;
            padding: 15px 0 20px 30px;
        }

        header img {
            width: 120px;
        }

        .hero-section {
            background: linear-gradient(to bottom, #CDE8FC, #EEF6FE);
            border-radius: 0 0 20px 20px;
            height: auto;
            padding: 0;
            position: relative;
        }

        .judul {
            text-align: center;
            margin: 60px 0 30px;
        }

        .judul h1 {
            font-size: 32px;
            font-weight: 700;
            margin-bottom: 10px;
        }

        .judul p {
            font-size: 18px;
            font-weight: 300;
            margin: 0;
        }

        .konten {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 30px;
            padding-bottom: 60px;
        }

        .card-role {
            display: flex;
            align-items: center;
            justify-content: space-between;
            width: 100%;
            max-width: 500px;
            padding: 20px 25px;
            border-radius: 18px;
            text-decoration: none;
            transition: all 0.3s ease;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.05);
        }

        .card-role:hover {
            transform: translateY(-4px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.08);
        }

        .card-role .text {
            max-width: 70%;
        }

        .card-role h2 {
            font-size: 20px;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .card-role p {
            font-size: 14px;
            font-weight: 300;
            margin: 0;
        }

        .card-role img {
            width: 70px;
            height: auto;
        }

        .mitra {
            background-color: #D9FBE2;
            color: #256d3d;
        }

        .alumni {
            background-color: #D4F0FF;
            color: #437F9F;
        }

        .siswa {
            background-color: #ece8ff;
            color: #5a4895;
        }

        a.text-decoration-none:hover {
            color: inherit;
        }

        @media (max-width: 768px) {
            .card-role {
                flex-direction: column;
                text-align: center;
                gap: 10px;
            }

            .card-role .text {
                max-width: 100%;
            }

            .card-role img {
                margin-top: 10px;
            }
        }

        .footer {
            background-color: #F2F9FF;
        }

        .logo-pu {
            min-width: 200px;
        }


        .navbar-brand img {
            height: 40px;
            object-fit: contain;
        }

        .navbar .img-fluid {
            max-height: 44px;
            object-fit: contain;
        }

        .tracking-wide {
            letter-spacing: 1px;
        }

        /* Responsive logo */
        .navbar-brand img {
            height: 55px;
            object-fit: contain;
        }

        .partner-logo {
            height: 34px;
        }

        /* Perbaiki tata letak partner logo di mobile */
        @media (max-width: 768px) {
            nav .navbar-brand {
                width: 100%;
                display: flex;
                justify-content: center;
                margin-bottom: 10px;
            }

            nav .d-flex.align-items-center {
                justify-content: center !important;
                flex-wrap: wrap;
                gap: 0.5rem;
                text-align: center;
            }

            nav .tracking-wide {
                font-size: 12px;
                width: 100%;
                margin-bottom: 5px;
            }

            nav .navbar .container {
                flex-direction: column;
                align-items: center;
            }

            .navbar-brand img {
                height: 50px !important;
            }

            .navbar .img-fluid {
                height: 26px !important;
            }

            .partner-section {
                align-items: center;
                justify-content: center;
                display: grid
            }

            .partner-logo {
                height: 26px;
            }
        }

        .partner-footer-logo {
            height: 40px;
            object-fit: contain;
        }

        .footer ul {
            padding-left: 0;
        }

        .footer li {
            margin-bottom: 8px;
        }

        .footer .bg-opacity-50 {
            background-color: rgba(255, 255, 255, 0.75) !important;
            /* semi-transparent white for readability */
        }

        .container-footer {
            width: 100%;
            height: 100%;
            ;
        }
    </style>
</head>

<body>
    <div class="body-utama">
        <!-- Navbar -->
        <nav class="navbar navbar-expand-lg" style="background: #CDE8FC">
            <div class="container">
                <a class="navbar-brand" href="#">
                    <img src="{{ url('img/logo.png') }}" alt="Mitraskul Logo" height="52">
                </a>
            </div>
        </nav>

        <div class="hero-section d-flex align-items-stretch p-0 position-relative">
            <div class="container-fluid d-flex align-items-stretch p-0">
                <div class="col-lg-1"></div>
                <div class="row w-100 m-0 align-items-stretch">
                    <div
                        class="col-lg-6 d-flex flex-column justify-content-center text-lg-start text-center py-5 px-4 ml-5">
                        <!-- ❌ Hapus logo di sini karena sudah dipindahkan ke atas -->
                        <!-- <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id" style="height: 50px;"> -->

                        <div>
                            <h1 class="fw-bold text-danger mb-3 ">Selamat Datang di <span
                                    class="text-primary">Mitraskul.id</span></h1>
                            <p class="fs-5 text-secondary">
                                Tempat terhubungnya alumni, sekolah, dunia kerja, dan peluang karier secara langsung dan
                                terpercaya.
                            </p>
                        </div>

                        <!-- Logo Partner -->
                        <div class="mt-4 partner-section">
                            <p class="text-muted mb-2 fw-semibold">Didukung oleh:</p>
                            <div class="d-flex align-items-center gap-3 flex-wrap">
                                <img src="{{ url('img/pu.png') }}" alt="Logo PU" class="partner-logo">
                                <img src="{{ url('img/telkomsel.png') }}" alt="Logo Telkomsel" class="partner-logo">
                            </div>
                        </div>
                    </div>

                    <div class="col-lg-6 d-flex align-items-end justify-content-center p-0">
                        <img src="{{ url('img/landing-logo.png') }}" alt="Ikon Utama"
                            style="width: 250px; margin-bottom: 0;">
                    </div>
                </div>
            </div>
        </div>


        <div class="judul">
            <h1>Ayo Mulai</h1>
            <p>Silahkan pilih role kamu:</p>
        </div>

        <!-- Role Cards -->
        <div class="container pb-5">
            <div class="row g-4 justify-content-center">
                <!-- Mitra -->
                <div class="col-md-6 col-lg-4 d-flex">
                    <a href="{{ route('login.mitra') }}" class="card-role mitra text-decoration-none w-100">
                        <div>
                            <h2>Mitra</h2>
                            <p>Bergabung menjadi mitra sebagai penyedia sertifikasi, pelatihan dan informasi lowongan
                                kerja</p>
                        </div>
                    </a>
                </div>

                <!-- Alumni -->
                <div class="col-md-6 col-lg-4 d-flex">
                    <a href="{{ route('login.alumni') }}" class="card-role alumni text-decoration-none w-100">
                        <div>
                            <h2>Alumni</h2>
                            <p>Jika kamu sudah lulus program pendidikan SMA/SMK/MA</p>
                        </div>
                    </a>
                </div>

                <!-- Siswa -->
                <div class="col-md-6 col-lg-4 d-flex">
                    <a href="{{ route('login.alumni') }}" class="card-role siswa text-decoration-none w-100">
                        <div>
                            <h2>Siswa</h2>
                            <p>Jika kamu sedang menempuh pendidikan di SMA/SMK/MA dan ingin mulai menjelajahi peluang
                                karier</p>
                        </div>
                    </a>
                </div>
            </div>
        </div>
    </div>
    <!-- Footer -->
    <footer class="footer text-dark"
        style="background: url('{{ url('img/footer.png') }}') no-repeat center center / cover;">
        <div class="container-footer">
            <div class="row align-items-start bg-white bg-opacity-75 rounded-3 p-4 shadow-sm">
                <!-- Logo & Deskripsi -->
                <div class="col-md-4 mb-4">
                    <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id" style="height: 65px;" class="mb-3">
                    <p class="text">
                        mitraskul.Id adalah platform yang menghubungkan alumni dengan sekolah, dunia industri,
                        dan
                        peluang karier.
                    </p>
                </div>

                <!-- Kontak -->
                <div class="col-md-4 mb-4">
                    <h6 class="fw-bold text-uppercase mb-3">Kontak</h6>
                    <ul class="list-unstyled small">
                        <li><i class="bi bi-envelope-fill me-2"></i>mitraskulid@gmail.com</li>
                        <li><i class="bi bi-telephone-fill me-2"></i>+62 851-7959-2408</li>
                    </ul>
                </div>

                <!-- Partner -->
                <div class="col-md-4 mb-4">
                    <h6 class="fw-bold text-uppercase mb-3">Didukung Oleh</h6>
                    <div class="d-flex align-items-center gap-3 flex-wrap">
                        <img src="{{ url('img/Telkomsel.png') }}" alt="Telkomsel" class="partner-footer-logo">
                        <img src="{{ url('img/pu.png') }}" alt="PUP Makassar" class="partner-footer-logo">
                    </div>
                </div>
                <div class="text-center text-dark small mt-4">
                    © 2025 mitraskul.Id. All rights reserved.
                </div>
            </div>

        </div>
    </footer>

</body>

</html>
