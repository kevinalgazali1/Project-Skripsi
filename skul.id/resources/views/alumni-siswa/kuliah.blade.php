<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">

    <style>
        /* ============ Base ============ */
        body {
            font-family: 'Poppins', sans-serif;
            margin: 0;
            background-color: #f8f9fa;
            overflow-x: hidden;
        }

        a {
            text-decoration: none;
        }

        .container-fluid {
            padding: 0;
            margin: 0;
        }

        /* ============ Navbar ============ */
        .navbar {
            background-color: #ffffff;
            padding: 1rem 2rem;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navbar .logo {
            width: 80px;
        }

        .user-info {
            display: flex;
            align-items: center;
        }

        .user-info span {
            margin-right: 1rem;
            font-weight: 600;
        }

        .user-info .profile-picture {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid #c04055;
        }

        /* ============ Layout ============ */
        .main-wrapper {
            display: flex;
            height: calc(100vh - 80px);
        }

        .artikel-card {
            display: flex;
            gap: 1rem;
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 10px;
            height: 100%;
        }

        .artikel-card img {
            width: 80px;
            height: 80px;
            border-radius: 8px;
            object-fit: cover;
            flex-shrink: 0;
        }

        .artikel-card .artikel-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }


        .sidebar {
            width: 220px;
            background-color: #eff9ff;
            padding: 2rem 2rem;
            flex-shrink: 0;
        }

        .sidebar a {
            display: block;
            color: #000;
            padding: 0.75rem 0;
            font-weight: 600;
        }

        .sidebar a:hover {
            color: #d24c62;
        }

        .sidebar .logout {
            margin-top: 2rem;
            color: #d24c62;
        }

        .content {
            flex: 1;
            overflow-y: auto;
            background-color: #fff;
            margin: 0;
        }

        /* ============ Banner ============ */
        .banner {
            background: url('{{ url('img/background.jpg') }}') no-repeat center center;
            padding: 2rem;
            position: relative;
            margin-bottom: 2rem;
            min-height: 350px;
        }

        .banner h2 {
            color: #c04055;
            font-weight: 700;
        }

        .banner p {
            max-width: 500px;
        }

        .banner .illustration {
            position: absolute;
            bottom: 0;
            right: 2rem;
            width: 350px;
        }

        /* ============ Fitur Section ============ */
        #fitur {
            max-width: 1000px;
            margin: 2rem auto;
        }

        .feature-card {
            background-color: #f1f1f1;
            border-radius: 15px;
            padding: 1.25rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: transform 0.2s ease;
        }

        .feature-card:hover {
            transform: scale(1.03);
        }

        .feature-content {
            flex: 1;
        }

        .feature-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .feature-card p {
            font-size: 0.95rem;
        }

        /* ============ Custom Feature Colors ============ */
        .sertifikasi {
            background-color: #DAF2FF;
        }

        .sertifikasi .feature-title {
            color: #196BE7;
        }

        .sertifikasi p {
            color: #428EFF;
        }

        .loker {
            background-color: #9BFFB9;
        }

        .loker .feature-title {
            color: #00924E;
        }

        .loker p {
            color: #00C569;
        }

        .pelatihan {
            background-color: #81F082;
        }

        .pelatihan .feature-title {
            color: #007B1F;
        }

        .pelatihan p {
            color: #00A92A;
        }

        .alumni {
            background-color: #F6C371;
        }

        .alumni .feature-title {
            color: #91641B;
        }

        .alumni p {
            color: #B98128;
        }

        .magang {
            background-color: #D9D9D9;
        }

        .magang .feature-title {
            color: #7D7D7D;
        }

        .magang p {
            color: #8E8E8E;
        }

        .footer {
            margin-top: 200px;
        }

        /* ============ Responsive ============ */
        @media (max-width: 768px) {
            .main-wrapper {
                flex-direction: column;
            }

            .sidebar {
                display: none;
            }

            .user-info {
                display: none;
                ;
            }

            .banner-images {
                flex-direction: column;
                align-items: flex-start;
            }

            .banner-images img {
                width: 100%;
            }

            .banner .illustration {
                display: none;
            }
        }

        @media (max-width: 1000px) {
            .main-wrapper {
                flex-direction: column;
            }

            .sidebar {
                display: none;
            }

            #fitur {
                margin-left: 20px;
                margin-right: 20px;
            }

            .banner-images {
                flex-direction: column;
                align-items: flex-start;
            }

            .banner-images img {
                width: 100%;
            }

            .banner .illustration {
                display: none;
            }
        }
    </style>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="col-lg-6 d-flex gap-4">
            <div class="col-lg-2">
                <img src="{{ url('img/logo.png') }}" alt="Logo" class="logo" width="" />
            </div>
            {{-- <div class="col-lg-2">
                <img src="{{ url('img/pu.jpeg') }}" alt="Logo" class="logo" width="" />
            </div> --}}
        </div>
        <div class="user-info">
            <span>Halo, {{ $user->alumniSiswaProfile->nama_lengkap }}</span>
            <img src="{{ asset('storage/' . $user->alumniSiswaProfile->foto_profil) }}" alt="Profile"
                class="profile-picture" />
        </div>
        <button class="btn d-md-none" type="button" data-bs-toggle="offcanvas" data-bs-target="#offcanvasSidebar">
            <i class="bi bi-list fs-3"></i>
        </button>
    </div>

    <div class="offcanvas offcanvas-start" tabindex="-1" id="offcanvasSidebar">
        <div class="offcanvas-header">
            <h5 class="offcanvas-title">Menu</h5>
            <button type="button" class="btn-close" data-bs-dismiss="offcanvas"></button>
        </div>
        <div class="offcanvas-body">
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.index') }}"><i
                    class="bi bi-house-door-fill me-2"></i>Beranda</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.sertifikasi') }}"><i
                    class="bi bi-patch-check-fill me-2"></i>Sertifikasi</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.loker') }}"><i
                    class="bi bi-briefcase-fill me-2"></i>Loker</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.pelatihan') }}"><i
                    class="bi bi-journal-text me-2"></i>Pelatihan</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('alumni-siswa.profile') }}"><i
                    class="bi bi-person-fill me-2"></i>Profil</a>
            <form id="logout-form" action="{{ route('logout') }}" method="POST">
                @csrf
                <a href="#" class="logout"
                    onclick="event.preventDefault(); document.getElementById('logout-form').submit();">Logout</a>
            </form>
        </div>
    </div>

    <div class="main-wrapper">
        <div class="sidebar">
            <a class="text-secondary" href="{{ route('alumni-siswa.index') }}"><i
                    class="bi bi-house-door-fill me-2"></i>Beranda</a>
            <a class="text-secondary" href="{{ route('alumni-siswa.sertifikasi') }}"><i
                    class="bi bi-patch-check-fill me-2"></i>Sertifikasi</a>
            <a class="text-secondary" href="{{ route('alumni-siswa.loker') }}"><i
                    class="bi bi-briefcase-fill me-2"></i>Loker</a>
            <a class="text-secondary" href="{{ route('alumni-siswa.pelatihan') }}"><i
                    class="bi bi-journal-text me-2"></i>Pelatihan</a>
            <a class="text-secondary" href="{{ route('alumni-siswa.profile') }}"><i
                    class="bi bi-person-fill me-2"></i>Profil</a>
            <form id="logout-form" action="{{ route('logout') }}" method="POST">
                @csrf
                <a href="#" class="logout"
                    onclick="event.preventDefault(); document.getElementById('logout-form').submit();">Logout</a>
            </form>
        </div>

        <div class="content">
            <div class="banner">
                <div
                    class="col-lg-7 d-flex flex-column justify-content-center text-lg-start text-center py-5 px-4 ml-5">
                    <h2 class="fw-bold">Bangun Masa Depanmu Bersama Skul.Id</h2>
                    <p class="text-secondary">Temukan peluang terbaik seperti pelatihan, sertifikasi, dan informasi
                        kampus untuk membantumu meraih cita-cita karier.</p>
                </div>
                <img src="{{ url('img/main-dashboard.png') }}" alt="Ilustrasi" class="illustration" />
            </div>

            <div class="container mt-5">
                <!-- Search Bar -->
                <div class="row mb-4 search-bar">
                    <div class="col-lg-12">
                        <input type="text" class="form-control shadow-sm"
                            placeholder="Cari nama kampus atau jurusan...">
                    </div>
                </div>

                <div class="row">
                    <!-- Daftar Kampus -->
                    <div class="col-lg-8 mb-4">
                        <div class="card p-4 border-0">

                            <!-- Kampus 1 -->
                            <div class="card mb-3 shadow-sm border-0 p-3 d-flex flex-row align-items-center">
                                <img src="https://www.ui.ac.id/wp-content/uploads/2021/02/logo-ui-2020.png"
                                    class="me-3" alt="Logo UI" width="70" />
                                <div>
                                    <h6 class="fw-semibold mb-1">Universitas Indonesia (UI)</h6>
                                    <p class="mb-1 text-muted">Kampus UI, Depok, Jawa Barat</p>
                                    <small class="text-secondary">(021) 7867222 | humas-ui@ui.ac.id</small>
                                </div>
                            </div>

                            <!-- Kampus 2 -->
                            <div class="card mb-3 shadow-sm border-0 p-3 d-flex flex-row align-items-center">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3e/Institut_Teknologi_Bandung_Logo.svg/800px-Institut_Teknologi_Bandung_Logo.svg.png"
                                    class="me-3" alt="Logo ITB" width="70" />
                                <div>
                                    <h6 class="fw-semibold mb-1">Institut Teknologi Bandung (ITB)</h6>
                                    <p class="mb-1 text-muted">Jl. Ganesha No.10, Bandung, Jawa Barat</p>
                                    <small class="text-secondary">(022) 2500935 | info@itb.ac.id</small>
                                </div>
                            </div>

                            <!-- Kampus 3 -->
                            <div class="card mb-3 shadow-sm border-0 p-3 d-flex flex-row align-items-center">
                                <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/Logo_UGM.png/600px-Logo_UGM.png"
                                    class="me-3" alt="Logo UGM" width="70" />
                                <div>
                                    <h6 class="fw-semibold mb-1">Universitas Gadjah Mada (UGM)</h6>
                                    <p class="mb-1 text-muted">Bulaksumur, Yogyakarta</p>
                                    <small class="text-secondary">(0274) 6491919 | info@ugm.ac.id</small>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Jurusan Favorit -->
                    <div class="col-lg-4 mb-4">
                        <div class="card p-4 shadow-sm border-0">
                            <h5 class="mb-3 fw-bold">Jurusan Favorit</h5>

                            <div class="card mb-3 shadow-sm border-0 p-3 card-jurusan">
                                <h6 class="fw-semibold mb-1">Kedokteran</h6>
                                <p class="mb-1 text-muted">Universitas Indonesia</p>
                                <small>Daya Tampung: 100 | Peminat: 3.000</small>
                            </div>

                            <div class="card mb-3 shadow-sm border-0 p-3 card-jurusan">
                                <h6 class="fw-semibold mb-1">Teknik Informatika</h6>
                                <p class="mb-1 text-muted">Institut Teknologi Bandung</p>
                                <small>Daya Tampung: 120 | Peminat: 4.200</small>
                            </div>

                            <div class="card mb-3 shadow-sm border-0 p-3 card-jurusan">
                                <h6 class="fw-semibold mb-1">Hukum</h6>
                                <p class="mb-1 text-muted">Universitas Gadjah Mada</p>
                                <small>Daya Tampung: 150 | Peminat: 3.800</small>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
</body>

</html>
