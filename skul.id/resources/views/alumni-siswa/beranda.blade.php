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

        .sidebar h5 {
            letter-spacing: 0.5px;
        }

        .logout {
            cursor: pointer;
        }


        .content {
            flex: 1;
            overflow-y: auto;
            background-color: #fff;
            overflow-x: hidden
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

        /* ============ Responsive ============ */
        @media (max-width: 768px) {
            .main-wrapper {
                flex-direction: column;
            }

            .sidebar-main {
                display: none !important;
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

        .article-content {
            line-height: 1.7;
            font-size: 1rem;
            text-align: justify;
            white-space: pre-wrap;
            /* Tambahkan ini */
        }

        .modal-article-content h3 {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }

        .modal-article-content .text-muted small {
            font-size: 0.9rem;
        }

        .modal-article-content .article-content {
            line-height: 1.7;
            font-size: 1rem;
            text-align: justify;
        }

        .modal-article-content img {
            max-height: 300px;
            object-fit: cover;
            border-radius: 12px;
            margin-bottom: 20px;
            width: 100%;
            max-width: 100%;
        }
    </style>
</head>

<body>
    <!-- Navbar -->
    <div class="navbar">
        <div class="col-lg-6 d-flex">
            <div class="col-lg-2">
                <img src="{{ url('img/logo.png') }}" alt="Logo" class="logo" width="" />
            </div>
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
                <a href="#" class="logout" onclick="event.preventDefault(); confirmLogout();">
            </form>


        </div>
    </div>

    <div class="main-wrapper">
        <div class="sidebar sidebar-main d-flex flex-column p-3" style="width: 250px;">


            <!-- Navigation Links -->
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.index') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.index') }}">
                <i class="bi bi-house-door-fill me-2"></i> Beranda
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.sertifikasi') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.sertifikasi') }}">
                <i class="bi bi-patch-check-fill me-2"></i> Sertifikasi
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.loker') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.loker') }}">
                <i class="bi bi-briefcase-fill me-2"></i> Loker
            </a>
            <a class="nav-link d-flex align-items-center mb-2 {{ request()->routeIs('alumni-siswa.pelatihan') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.pelatihan') }}">
                <i class="bi bi-journal-text me-2"></i> Pelatihan
            </a>
            <a class="nav-link d-flex align-items-center mb-4 {{ request()->routeIs('alumni-siswa.profile') ? 'active' : 'text-secondary' }}"
                href="{{ route('alumni-siswa.profile') }}">
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

            <div class="container-fluid mt-5">
                <div id="fitur">
                    <h5 class="mb-4 fw-semibold">Fitur Utama :</h5>
                    <div class="row gx-4 gy-4 mb-4">
                        <div class="col-md-4">
                            <a href="{{ route('alumni-siswa.sertifikasi') }}">
                                <div class="feature-card sertifikasi h-100">
                                    <div class="feature-content">
                                        <div class="feature-title">Sertifikasi</div>
                                        <p>Perkuat kredensialmu dengan sertifikasi profesional.</p>
                                    </div>
                                </div>
                            </a>
                        </div>
                        <div class="col-md-4">
                            <a href="{{ route('alumni-siswa.loker') }}">
                                <div class="feature-card loker h-100">
                                    <div class="feature-content">
                                        <div class="feature-title">Loker</div>
                                        <p>Temukan pekerjaan yang sesuai dengan minat dan keahlianmu.</p>
                                    </div>
                                </div>
                            </a>
                        </div>
                        <div class="col-md-4">
                            <a href="{{ route('alumni-siswa.pelatihan') }}">
                                <div class="feature-card pelatihan h-100">
                                    <div class="feature-content">
                                        <div class="feature-title">Pelatihan</div>
                                        <p>Ikuti pelatihan untuk meningkatkan kompetensimu.</p>
                                    </div>
                                </div>
                            </a>
                        </div>
                    </div>

                    <div class="row gx-4 gy-4">
                        <div class="col-md-6">
                            <a href="{{ route('alumni-siswa.ikatan') }}">
                                <div class="feature-card alumni h-100">
                                    <div class="feature-content">
                                        <div class="feature-title">Ikatan Alumni</div>
                                        <p>Bangun koneksi dengan sesama alumni dari sekolahmu.</p>
                                    </div>
                                </div>
                            </a>
                        </div>
                        {{-- <div class="col-md-6">
                            <a href="{{ route('alumni-siswa.kuliah') }}">
                                <div class="feature-card magang h-100">
                                    <div class="feature-content">
                                        <div class="feature-title">Kuliah</div>
                                        <p>Informasi dan panduan kuliah di berbagai universitas Indonesia.</p>
                                    </div>
                                </div>
                            </a>
                        </div> --}}
                    </div>
                    <h5 class="mb-4 mt-5 fw-semibold">Fitur Add-On :</h5>
                    <div class="row gx-4 gy-4 mb-4 text-center">
                        <div class="col-6 col-md-2">
                            <a href="{{ route('alumni-siswa.artikel') }}" class="text-decoration-none text-dark">
                                <div>
                                    <i class="bi bi-file-earmark-text-fill fs-1 text-danger"></i>
                                    <div class="mt-2 fw-semibold">Artikel</div>
                                </div>
                            </a>
                        </div>
                        <div class="col-6 col-md-2">
                            <a href="https://www.telkomsel.com/shops/digital-product/package?utm_source=wec&utm_medium=skulbanner&utm_campaign=EPskul&utm_id=EPskul&embed=on&msisdn=&type=Kamu%20Banget&category=&roaming=&service=&sort=lowest-price&minPrice=&maxPrice=&quota="
                                class="text-decoration-none text-dark" target="_blank">
                                <div>
                                    <i class="bi bi-wifi fs-1 text-info"></i>
                                    <div class="mt-2 fw-semibold">Pulsa & Internet</div>
                                </div>
                            </a>
                        </div>
                        <div class="col-6 col-md-2">
                            <a href="https://account.schoters.com/en/login?redirect_url=https%3A%2F%2Facademy.schoters.com%2Fdemo%2Fclaim%3Fcode%3Dschotersskulid&client_id=schoters&view=login"
                                class="text-decoration-none text-dark" target="_blank">
                                <div>
                                    <i class="bi bi-mortarboard-fill fs-1 text-success"></i>
                                    <div class="mt-2 fw-semibold">Beasiswaku</div>
                                </div>
                            </a>
                        </div>
                    </div>
                    <!-- Voucher Games Section -->
                    <div class="container-fluid my-5">
                        <h5 class="text-danger fw-semibold py-1 mt-10">
                            <i class="bi bi-controller me-2"></i>VOUCHER GAMES
                        </h5>
                        <p class="text-muted">Upgrade karakter game kamu dengan beli vouchernya!</p>
                        <div class="d-flex flex-wrap align-items-center gap-4 mt-3 mb-5">
                            <a href="https://duniagames.co.id/top-up/item/mobile-legends" target="_blank">
                                <img src="img/mlbb.png" alt="Mobile Legends" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                            <a href="https://duniagames.co.id/top-up/item/call-of-duty-mobile" target="_blank">
                                <img src="img/codm.png" alt="COD Mobile" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                            <a href="https://duniagames.co.id/top-up/item/razer-gold" target="_blank">
                                <img src="img/zepeto.png" alt="Zepeto Gold" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                            <a href="https://duniagames.co.id/top-up/item/google-play" target="_blank">
                                <img src="img/googleplay.png" alt="Google Play" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                            <a href="https://duniagames.co.id/top-up/item/aov" target="_blank">
                                <img src="img/aov.png" alt="AOV" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                            <a href="https://duniagames.co.id/top-up/item/freefire" target="_blank">
                                <img src="img/freefire.png" alt="Free Fire" class="rounded-circle"
                                    style="width: 80px; height: 80px;">
                            </a>
                        </div>
                    </div>


                    <!-- Artikel Section -->
                    <div class="d-flex justify-content-between align-items-center mb-3">
                        <h5 class="text-danger fw-semibold">
                            <i class="bi bi-journal-richtext me-2 py-1 mt-10"></i>ARTIKEL
                        </h5>
                        <a href="{{ route('alumni-siswa.artikel') }}"
                            class="text-danger text-decoration-none fw-semibold">Lihat Semua</a>
                    </div>

                    <div class="row g-4 mb-5">
                        @foreach ($artikels as $artikel)
                            <div class="col-md-6 col-xl-4">
                                <div class="card h-100 shadow-sm border-0 rounded-4 overflow-hidden">
                                    <img src="{{ asset('storage/' . $artikel->gambar_artikel) }}"
                                        onerror="this.src='https://my.skul.id/static/media/img_default_artikel_thumb.03a53a33.svg'"
                                        class="card-img-top" alt="Thumbnail Artikel"
                                        style="height: 200px; object-fit: cover;">

                                    <div class="card-body d-flex flex-column">
                                        <h5 class="card-title fw-semibold mb-2" style="min-height: 3rem;">
                                            {{ \Illuminate\Support\Str::limit($artikel->judul, 50) }}
                                        </h5>
                                        <p class="text-muted small mb-3">Oleh {{ $artikel->penulis }}</p>

                                        <div class="mt-auto">
                                            <button class="btn btn-sm btn-outline-primary rounded-pill px-3"
                                                onclick="openDetailArtikelFromCard(
                                '{{ $artikel->judul }}',
                                '{{ $artikel->penulis }}',
                                '{{ $artikel->created_at->format('d M Y') }}',
                                `{!! htmlentities($artikel->isi) !!}`,
                                '{{ asset('storage/' . $artikel->gambar_artikel) }}'
                            )">
                                                <i class="bi bi-eye me-1"></i> Baca
                                            </button>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        @endforeach
                    </div>

                    <div class="modal fade" id="lihatArtikelModal" tabindex="-1"
                        aria-labelledby="lihatArtikelModalLabel" aria-hidden="true">
                        <div class="modal-dialog modal-xl modal-dialog-scrollable">
                            <div class="modal-content border-0 rounded-4 modal-article-content">
                                <div class="modal-header border-0">
                                    <h5 class="modal-title fw-semibold" id="lihatArtikelModalLabel">Detail Artikel
                                    </h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"
                                        aria-label="Tutup"></button>
                                </div>
                                <div class="modal-body px-4 pb-4">
                                    <div id="detailGambar" class="mb-4 text-center">
                                        <img src="" alt="Gambar Artikel" class="img-fluid rounded shadow-sm">
                                    </div>
                                    <h3 id="detailJudul" class="text-center"></h3>
                                    <div class="text-muted mb-4 text-center">
                                        <small>Ditulis oleh <span id="detailPenulis"></span> pada <span
                                                id="detailTanggal"></span></small>
                                    </div>
                                    <div id="detailIsi" class="article-content" style="white-space: pre-wrap;"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Footer -->
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
                                <img src="{{ url('img/Telkomsel.png') }}" alt="Telkomsel"
                                    class="partner-footer-logo">
                                <img src="{{ url('img/pu.png') }}" alt="PUP Makassar" class="partner-footer-logo">
                            </div>
                        </div>
                        <div class="text-center text-dark small mt-4">
                            © 2025 Skul.Id. All rights reserved.
                        </div>
                    </div>

                </div>
            </footer>



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
            <script>
                function openDetailArtikelFromCard(judul, penulis, tanggal, isi, gambar) {
                    document.getElementById('detailJudul').textContent = judul;
                    document.getElementById('detailPenulis').textContent = penulis;
                    document.getElementById('detailTanggal').textContent = tanggal;
                    document.getElementById('detailGambar').querySelector('img').src = gambar;

                    // Jika isi mengandung tag HTML: pakai innerHTML + decode
                    // Jika tidak mengandung tag HTML (plain text): cukup pakai textContent atau pre-wrap
                    document.getElementById('detailIsi').textContent = decodeHTMLEntities(isi);

                    const modal = new bootstrap.Modal(document.getElementById('lihatArtikelModal'));
                    modal.show();
                }

                function decodeHTMLEntities(text) {
                    const textarea = document.createElement('textarea');
                    textarea.innerHTML = text;
                    return textarea.value;
                }
            </script>

</html>
