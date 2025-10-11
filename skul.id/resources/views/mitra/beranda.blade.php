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
            overflow: hidden;
        }

        a,
        a.card-role {
            text-decoration: none;
        }

        .container-fluid {
            padding: 0;
            margin: 0;
        }

        /* ============ Navbar ============ */
        .navbar {
            background-color: #ffffff;
            border-bottom: 1px solid #ddd;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .navbar .logo {
            width: 80px;
            margin-left: 30px;
        }

        .user-info {
            display: flex;
            align-items: center;
            margin-right: 30px;
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

        /* ============ Sidebar ============ */
        .sidebar {
            font-family: 'Poppins', sans-serif;
            font-size: 18px;
            height: 100%;
            background-color: #eff9ff;
            position: relative;
            flex-shrink: 0;
        }

        .nav-link {
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            color: #495057;
            text-decoration: none;
            transition: background 0.2s, color 0.2s;
        }

        .nav-link.active {
            background-color: #e7f1ff;
            color: #0d6efd;
            font-weight: 600;
        }

        .nav-link i {
            font-size: 1.1rem;
        }

        .logout {
            cursor: pointer;
        }

        /* ============ Layout ============ */
        .main-wrapper {
            display: flex;
            height: calc(100vh - 80px);
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

        .fitur-box {
            border-radius: 15px;
            padding: 1.25rem;
            min-height: 160px;
            transition: transform 0.2s ease;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .fitur-box:hover {
            transform: scale(1.03);
        }

        .fitur-title {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .fitur-box p {
            font-size: 0.95rem;
            font-weight: 400;
        }

        .sertifikasi {
            background-color: #DAF2FF;
        }

        .sertifikasi .fitur-title {
            color: #196BE7;
        }

        .sertifikasi p {
            color: #428EFF;
        }

        .loker {
            background-color: #9BFFB9;
        }

        .loker .fitur-title {
            color: #00924E;
        }

        .loker p {
            color: #00C569;
        }

        .pelatihan {
            background-color: #81F082;
        }

        .pelatihan .fitur-title {
            color: #007B1F;
        }

        .pelatihan p {
            color: #00A92A;
        }

        .alumni {
            background-color: #F6C371;
        }

        .alumni .fitur-title {
            color: #91641B;
        }

        .alumni p {
            color: #B98128;
        }

        .magang {
            background-color: #D9D9D9;
        }

        .magang .fitur-title {
            color: #7D7D7D;
        }

        .magang p {
            color: #8E8E8E;
        }

        /* ============ Footer ============ */
        .footer {
            margin-top: 200px;
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
        }

        .container-footer {
            width: 100%;
            height: 100%;
        }

        /* ============ Responsive ============ */
        @media (max-width: 1000px) {
            .main-wrapper {
                flex-direction: column;
            }

            #fitur {
                margin-left: 20px;
                margin-right: 20px;
            }

            .banner-images {
                flex-direction: column;
                align-items: flex-start;
            }

            .banner-images img,
            .banner .illustration {
                display: none;
            }
        }
    </style>
</head>

<body>
    <div class="navbar">
        <div class="col-lg-6 d-flex">
            <div class="col-lg-2">
                <img src="{{ url('img/logo.png') }}" alt="Logo" class="logo" width="" />
            </div>
        </div>
        <div class="user-info">
            <span>Halo, {{ $user->mitraProfile->nama_instansi }}</span>
            <img src="{{ url('img/user.png') }}" alt="Profile" class="profile-picture" />
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
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('mitra.index') }}"><i
                    class="bi bi-house-door-fill me-2"></i>Beranda</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('mitra.sertifikasi') }}"><i
                    class="bi bi-patch-check-fill me-2"></i>Sertifikasi</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('mitra.loker') }}"><i
                    class="bi bi-briefcase-fill me-2"></i>Loker</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('mitra.pelatihan') }}"><i
                    class="bi bi-journal-text me-2"></i>Pelatihan</a>
            <a class="d-block mb-2 text-dark fw-semibold" href="{{ route('mitra.profile') }}"><i
                    class="bi bi-person-fill me-2"></i>Profil</a>
            <form id="logout-form" action="{{ route('logout') }}" method="POST">
                @csrf
                <a href="#" class="logout" onclick="event.preventDefault(); confirmLogout();">
            </form>
        </div>
    </div>

    <div class="main-wrapper">
        <div class="sidebar sidebar-main d-flex flex-column p-3" style="width: 250px;">


            <!-- Navigation Links -->
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('mitra.index') ? 'active' : 'text-secondary' }}"
                href="{{ route('mitra.index') }}">
                <i class="bi bi-house-door-fill me-2"></i> Beranda
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('mitra.sertifikasi') ? 'active' : 'text-secondary' }}"
                href="{{ route('mitra.sertifikasi') }}">
                <i class="bi bi-patch-check-fill me-2"></i> Sertifikasi
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('mitra.loker') ? 'active' : 'text-secondary' }}"
                href="{{ route('mitra.loker') }}">
                <i class="bi bi-briefcase-fill me-2"></i> Loker
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('mitra.pelatihan') ? 'active' : 'text-secondary' }}"
                href="{{ route('mitra.pelatihan') }}">
                <i class="bi bi-journal-text me-2"></i> Pelatihan
            </a>
            <a class="nav-link d-flex align-items-center mb-4 {{ request()->routeIs('mitra.profile') ? 'active' : 'text-secondary' }}"
                href="{{ route('mitra.profile') }}">
                <i class="bi bi-person-fill me-2"></i> Profil
            </a>

            <!-- Logout -->
            <form id="logout-form" action="{{ route('logout') }}" method="POST" class="mt-auto">
                @csrf
                <a href="#" class="nav-link d-flex align-items-center text-danger fw-semibold mb-4"
                    onclick="confirmLogout(event)">
                    <i class="bi bi-box-arrow-right me-2"></i> Logout
                </a>
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

            <div class="container my-5">
                <div id="fitur">
                    <h5 class="mb-4 text-primary">Fitur Utama :</h5>

                    <div class="row gx-4 gy-4 mb-4">
                        <div class="col-md-4">
                            <a href="{{ route('mitra.sertifikasi') }}" class="card-role">
                                <div class="fitur-box sertifikasi h-100">
                                    <div class="fitur-title">Sertifikasi</div>
                                    <p>Meningkatkan anda mencantumkan yang anda butuhkan</p>
                                </div>
                            </a>
                        </div>
                        <div class="col-md-4">
                            <a href="{{ route('mitra.loker') }}" class="card-role">
                                <div class="fitur-box loker h-100">
                                    <div class="fitur-title">Loker</div>
                                    <p>Temukan pekerjaan yang anda impikan</p>
                                </div>
                            </a>
                        </div>
                        <div class="col-md-4">
                            <a href="{{ route('mitra.pelatihan') }}" class="card-role">
                                <div class="fitur-box pelatihan h-100">
                                    <div class="fitur-title">Pelatihan</div>
                                    <p>Ikuti pelatihan untuk meningkatkan skill anda</p>
                                </div>
                            </a>
                        </div>
                    </div>
                </div>
            </div>
            <footer class="footer text-dark"
                style="background: url('{{ url('img/footer.png') }}') no-repeat center center / cover;">
                <div class="container-footer">
                    <div class="row align-items-start bg-white bg-opacity-75 rounded-3 p-4 shadow-sm">
                        <!-- Logo & Deskripsi -->
                        <div class="col-md-4 mb-4">
                            <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id" style="height: 65px;"
                                class="mb-3">
                            <p class="small">
                                mitraskul.Id adalah platform yang menghubungkan alumni dengan sekolah, dunia industri,
                                dan
                                peluang karier.
                            </p>
                        </div>

                        <!-- Kontak -->
                        <div class="col-md-4 mb-4">
                            <!-- <h6 class="fw-bold text-uppercase mb-3">Kontak</h6>
                            <ul class="list-unstyled small">
                                <li><i class="bi bi-envelope-fill me-2"></i>mitraskulid@gmail.com</li>
                                <li><i class="bi bi-telephone-fill me-2"></i>+62 851-7959-2408</li>
                            </ul> -->
                        </div>

                        <!-- Partner -->
                        <div class="col-md-4 mb-4">
                            <h6 class="fw-bold text-uppercase mb-3">Didukung Oleh</h6>
                            <div class="d-flex align-items-center gap-3 flex-wrap">
                                <img src="{{ url('img/Telkomsel.png') }}" alt="Telkomsel"
                                    class="partner-footer-logo">
                                <img src="{{ url('img/pu.png') }}" alt="PUP Makassar" class="partner-footer-logo">
                            </div>
                        </div>
                        <div class="text-center text-dark small mt-4">
                            © 2025 mitraskul.Id. All rights reserved.
                        </div>
                    </div>

                </div>
            </footer>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
        <script>
            function confirmLogout() {
                event.preventDefault();
                Swal.fire({
                    title: 'Yakin ingin logout?',
                    text: "Anda akan keluar dari akun ini.",
                    icon: 'warning',
                    showCancelButton: true,
                    confirmButtonColor: '#d33',
                    cancelButtonColor: '#6c757d',
                    confirmButtonText: 'Ya, logout!',
                    cancelButtonText: 'Batal'
                }).then((result) => {
                    if (result.isConfirmed) {
                        document.getElementById('logout-form').submit();
                    }
                });
            }
        </script>
</body>

</html>
