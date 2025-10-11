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
            overflow-x: hidden;
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

        .certificate-card {
            display: flex;
            flex-direction: column;
            background: #fff;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.06);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            max-width: 360px;
        }

        .certificate-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.08);
        }

        .certificate-img {
            width: 100%;
            height: 180px;
            object-fit: cover;
            padding-top: 10px;
        }

        .certificate-content {
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            height: 100%;
        }

        .certificate-title {
            font-size: 1rem;
            font-weight: 600;
            color: #212529;
            margin-bottom: 0.5rem;
        }

        .certificate-provider {
            font-size: 0.875rem;
            color: #6c757d;
            margin-bottom: 0.25rem;
        }

        .certificate-date {
            font-size: 0.8rem;
            color: #adb5bd;
            margin-bottom: 1rem;
        }

        .badge {
            font-size: 0.75rem;
            padding: 0.35em 0.6em;
            border-radius: 0.5rem;
        }

        .btn-outline-primary:hover {
            background-color: #0d6efd;
            color: white;
        }

        .hero-banner {
            min-height: 350px;
        }

        .secion-hl {
            background: url('{{ url('img/background.jpg') }}') no-repeat center center;
            padding: 4rem 1rem;
            width: 100%;
            min-height: 400px;
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

        .modal-content {
            animation: fadeInUp 0.3s ease-in-out;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .loker-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease;
        }

        .loker-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
        }

        .btn-view.secondary {
            background-color: #f3f4f6;
            color: #374151;
        }

        .btn-view.secondary:hover {
            background-color: #e5e7eb;
        }

        .btn-action {
            outline: none;
        }

        .loker-card.rekomendasi {
            border-left: 5px solid #0d6efd;
            transition: all 0.3s ease;
        }

        .loker-card.rekomendasi:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 15px rgba(0, 0, 0, 0.1);
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
                    <i class="bi bi-box-arrow-right me-2"></i>Logout</a>
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
                    onclick="event.preventDefault(); confirmLogout();">
                    <i class="bi bi-box-arrow-right me-2"></i> Logout
                </a>
            </form>
        </div>



        <div class="content">
            <div class="hero-banner p-5 mb-4 text-white rounded-4 shadow-sm"
                style="background: url('{{ url('img/background.jpg') }}') no-repeat center center;">
                <div class="row align-items-center section-hl">
                    <div class="text-center">
                        <h1 class="fw-bold mb-3 text-danger mt-4">Bangun Karier Impianmu 🚀</h1>
                        <p class="lead text-secondary">Jelajahi peluang kerja terbaik, pelatihan profesional, dan
                            sertifikasi untuk alumni seperti kamu.</p>
                    </div>
                </div>
            </div>

            <div class="main-content px-5 py-4">

                <div class="row g-4">
                    <form action="{{ route('alumni-siswa.loker.search') }}" method="GET"
                        class="row g-3 align-items-center mb-4">
                        <div class="col-md-3">
                            <input type="text" name="search" class="form-control"
                                placeholder="Cari posisi / perusahaan..." value="{{ request('search') }}">
                        </div>

                        <div class="col-md-3">
                            <select name="tipe" class="form-select">
                                <option value="">Semua Jenis</option>
                                <option value="full time" {{ request('tipe') == 'Full-time' ? 'selected' : '' }}>
                                    Full-time
                                </option>
                                <option value="part time" {{ request('tipe') == 'Part time' ? 'selected' : '' }}>Part
                                    time
                                </option>
                                <option value="magang" {{ request('tipe') == 'Magang' ? 'selected' : '' }}>Magang
                                </option>
                            </select>
                        </div>

                        <div class="col-md-3">
                            <select name="lokasi" class="form-select">
                                <option value="">Semua Lokasi</option>
                                @foreach ($lokasiList as $lokasi)
                                    <option value="{{ $lokasi }}"
                                        {{ request('lokasi') == $lokasi ? 'selected' : '' }}>
                                        {{ $lokasi }}
                                    </option>
                                @endforeach
                            </select>
                        </div>

                        <div class="col-md-3 d-flex gap-2">
                            <button type="submit" class="btn btn-primary w-100">Filter</button>
                            <a href="{{ route('alumni-siswa.loker') }}" class="btn btn-secondary">Reset</a>
                        </div>
                    </form>

                    {{-- Bagian 2: Rekomendasi Pekerjaan (tambahkan di sini) --}}
                    @if (isset($recommendations) && count($recommendations) > 0)
                        <div class="mb-5">
                            <h4 class="fw-bold mb-4 text-primary">
                                <i class="bi bi-stars me-2"></i> Rekomendasi Pekerjaan Untukmu
                            </h4>

                            <div class="row">
                                @foreach ($recommendations as $job)
                                    <div class="col-md-6 col-lg-4 mb-4">
                                        <div class="card shadow-sm h-100 border-1 rounded-4 loker-card hover-shadow transition"
                                            role="button" data-bs-toggle="modal"
                                            data-bs-target="#rekomModal{{ $job['id'] }}" style="cursor: pointer;">

                                            <div
                                                class="card-body p-4 d-flex flex-column justify-content-between h-100">
                                                <div class="d-flex align-items-center gap-3 mb-3">
                                                    <img src="{{ asset('storage/' . $job['gambar']) }}"
                                                        alt="Logo" class="rounded-circle border"
                                                        style="width: 55px; height: 55px; object-fit: cover;">
                                                    <div>
                                                        <h6 class="fw-semibold mb-1 text-dark">{{ $job['posisi'] }}
                                                        </h6>
                                                        <div class="text-muted small d-flex align-items-center">
                                                            <i class="bi bi-buildings me-1"></i>
                                                            {{ $job['nama_perusahaan'] }}
                                                        </div>
                                                    </div>
                                                </div>

                                                <div class="mb-2 text-muted small">
                                                    <i class="bi bi-geo-alt me-1 text-primary"></i>
                                                    {{ $job['lokasi'] }}
                                                </div>

                                                <div class="mb-3">
                                                    <span class="badge bg-light text-dark border me-1">
                                                        <i class="bi bi-mortarboard me-1"></i>{{ $job['pendidikan'] }}
                                                    </span>
                                                    <span class="badge bg-success text-white">
                                                        {{ number_format($job['relevansi_persen'], 2) }}% cocok
                                                    </span>
                                                </div>

                                                <div class="d-flex gap-2 mt-auto btn-action">
                                                    <button class="btn btn-primary btn-sm w-100"
                                                        data-bs-toggle="modal"
                                                        data-bs-target="#daftarRekomModal{{ $job['id'] }}"
                                                        onclick="event.stopPropagation();">
                                                        <i class="bi bi-send me-1"></i> Daftar
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {{-- Modal Detail Rekomendasi --}}
                                    <div class="modal fade" id="rekomModal{{ $job['id'] }}" tabindex="-1"
                                        aria-hidden="true">
                                        <div class="modal-dialog modal-lg modal-dialog-centered">
                                            <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                                                <div
                                                    class="p-4 bg-white border-bottom d-flex align-items-center justify-content-between">
                                                    <h5 class="fw-bold mb-0">{{ $job['posisi'] }}</h5>
                                                    <button type="button" class="btn-close" data-bs-dismiss="modal"
                                                        aria-label="Close"></button>
                                                </div>
                                                <div class="modal-body p-4">
                                                    <p><strong>Perusahaan:</strong> {{ $job['nama_perusahaan'] }}</p>
                                                    <p><strong>Lokasi:</strong> {{ $job['lokasi'] }}</p>
                                                    <p><strong>Pendidikan:</strong> {{ $job['pendidikan'] }}</p>
                                                    <p><strong>Kecocokan:</strong>
                                                        {{ number_format($job['relevansi_persen'], 2) }}%</p>
                                                    <p class="text-muted small fst-italic">
                                                        {{ $job['deskripsi'] }}
                                                    </p>
                                                </div>
                                                <div class="modal-footer bg-light border-0">
                                                    <button class="btn btn-primary" data-bs-toggle="modal"
                                                        data-bs-target="#daftarRekomModal{{ $job['id'] }}">
                                                        <i class="bi bi-send me-1"></i> Lamar Sekarang
                                                    </button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    {{-- Modal Daftar --}}
                                    <div class="modal fade" id="daftarRekomModal{{ $job['id'] }}" tabindex="-1"
                                        aria-hidden="true">
                                        <div class="modal-dialog modal-dialog-centered">
                                            <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                                                <div class="bg-primary text-white p-4">
                                                    <h5 class="modal-title mb-1">Lamar di
                                                        {{ $job['nama_perusahaan'] }}</h5>
                                                    <p class="mb-0 small">Posisi:
                                                        <strong>{{ $job['posisi'] }}</strong>
                                                    </p>
                                                </div>
                                                <form id="formRekomLamaran{{ $job['id'] }}"
                                                    action="{{ route('alumni-siswa.loker.store', $job['id']) }}"
                                                    method="POST" enctype="multipart/form-data">
                                                    @csrf
                                                    <div class="modal-body p-4">
                                                        <div class="mb-4">
                                                            <label for="cvRekom{{ $job['id'] }}"
                                                                class="form-label fw-semibold">
                                                                Unggah CV Anda <span class="text-danger">*</span>
                                                            </label>
                                                            <input type="file" name="cv"
                                                                id="cvRekom{{ $job['id'] }}" class="form-control"
                                                                accept=".pdf" required>
                                                            <div class="form-text">Hanya format PDF. Maksimal 2MB.
                                                            </div>
                                                        </div>
                                                    </div>
                                                    <div
                                                        class="modal-footer bg-light border-0 d-flex justify-content-between px-4 py-3">
                                                        <button type="button" class="btn btn-outline-secondary"
                                                            data-bs-dismiss="modal">Batal</button>
                                                        <button type="button" class="btn btn-primary"
                                                            onclick="confirmKirimLamaran('formRekomLamaran{{ $job['id'] }}')">
                                                            <i class="bi bi-send me-1"></i> Kirim Lamaran
                                                        </button>
                                                    </div>
                                                </form>
                                            </div>
                                        </div>
                                    </div>
                                @endforeach
                            </div>
                        </div>
                    @endif


                    {{-- Bagian 3: Semua Pekerjaan (tambahkan di sini) --}}
                    <h4 class="fw-bold mb-4 text-primary">
                        <i class="bi bi-stars me-2"></i> Semua Pekerjaan
                    </h4>
                    @foreach ($loker as $item)
                        <div class="col-md-6 col-lg-4 mb-4">
                            <div class="card shadow-sm h-100 border-1 rounded-4 loker-card hover-shadow transition"
                                role="button" data-bs-toggle="modal"
                                data-bs-target="#detailModal{{ $item->id }}" style="cursor: pointer;">

                                <div class="card-body p-4 d-flex flex-column justify-content-between h-100">

                                    <!-- Header: Logo & Nama -->
                                    <div class="d-flex align-items-center gap-3 mb-3">
                                        <img src="{{ asset('storage/' . $item->gambar) }}" alt="Logo"
                                            class="rounded-circle border"
                                            style="width: 55px; height: 55px; object-fit: cover;">
                                        <div>
                                            <h6 class="fw-semibold mb-1 text-dark">{{ $item->posisi }}</h6>
                                            <div class="text-muted small d-flex align-items-center">
                                                <i class="bi bi-buildings me-1"></i> {{ $item->nama_perusahaan }}
                                            </div>
                                        </div>
                                    </div>

                                    <!-- Info Lokasi & Jenis -->
                                    <div class="mb-2 text-muted small">
                                        <i class="bi bi-geo-alt me-1 text-primary"></i> {{ $item->lokasi }} •
                                        {{ $item->tipe }}
                                    </div>

                                    <!-- Badge Pendidikan & Gaji -->
                                    <div class="mb-3">
                                        <span class="badge bg-light text-dark border me-1">
                                            <i class="bi bi-mortarboard me-1"></i>{{ $item->pendidikan }}
                                        </span>
                                        <span class="badge bg-light text-dark border">
                                            {{ $item->gaji }}
                                        </span>
                                    </div>


                                    <!-- Action Button -->
                                    <div class="d-flex gap-2 mt-auto btn-action">
                                        <button class="btn btn-primary btn-sm w-100" data-bs-toggle="modal"
                                            data-bs-target="#daftarModal{{ $item->id }}"
                                            onclick="event.stopPropagation();">
                                            <i class="bi bi-send me-1"></i> Daftar
                                        </button>
                                    </div>

                                </div>
                            </div>
                        </div>




                        {{-- Modal Detail --}}
                        <div class="modal fade" id="detailModal{{ $item->id }}" tabindex="-1"
                            aria-hidden="true">
                            <div class="modal-dialog modal-xl modal-dialog-centered">
                                <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                                    <!-- Header -->
                                    <div
                                        class="p-4 bg-white border-bottom d-flex flex-column flex-md-row align-items-center gap-4">
                                        <!-- Logo Perusahaan -->
                                        <div class="flex-shrink-0">
                                            <img src="{{ asset('storage/' . $item->gambar) }}"
                                                alt="Logo {{ $item->nama_perusahaan }}"
                                                class="rounded-3 border shadow-sm"
                                                style="width: 90px; height: 90px; object-fit: cover;">
                                        </div>

                                        <!-- Info Posisi & Perusahaan -->
                                        <div class="flex-grow-1 text-center text-md-start">
                                            <h3 class="fw-bold mb-1 text-dark">{{ $item->posisi }}</h3>
                                            <div
                                                class="text-muted fs-6 d-flex justify-content-center justify-content-md-start align-items-center gap-2">
                                                <i class="bi bi-buildings"></i>
                                                <span>{{ $item->nama_perusahaan }}</span>
                                            </div>
                                        </div>

                                        <!-- Tombol Close -->
                                        <button type="button" class="btn-close ms-auto" data-bs-dismiss="modal"
                                            aria-label="Close"></button>
                                    </div>


                                    <!-- Body -->
                                    <div class="modal-body p-4">
                                        <!-- Informasi Umum -->
                                        <div class="row g-3 mb-4">
                                            <div class="col-md-6 d-flex align-items-start">
                                                <i class="bi bi-geo-alt-fill text-primary me-3 fs-4"></i>
                                                <div>
                                                    <strong class="d-block">Lokasi</strong>
                                                    <span class="text-muted">{{ $item->lokasi }}</span>
                                                </div>
                                            </div>
                                            <div class="col-md-6 d-flex align-items-start">
                                                <i class="bi bi-briefcase-fill text-primary me-3 fs-4"></i>
                                                <div>
                                                    <strong class="d-block">Jenis Pekerjaan</strong>
                                                    <span class="text-muted">{{ $item->tipe }}</span>
                                                </div>
                                            </div>
                                            <div class="col-md-6 d-flex align-items-start">
                                                <i class="bi bi-mortarboard-fill text-primary me-3 fs-4"></i>
                                                <div>
                                                    <strong class="d-block">Pendidikan Minimum</strong>
                                                    <span class="text-muted">{{ $item->pendidikan }}</span>
                                                </div>
                                            </div>
                                            <div class="col-md-6 d-flex align-items-start">
                                                <i class="bi bi-cash-stack text-primary me-3 fs-4"></i>
                                                <div>
                                                    <strong class="d-block">Gaji</strong>
                                                    <span class="text-muted">{{ $item->gaji }}</span>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Deskripsi -->
                                        <div>
                                            <h5 class="fw-semibold mb-2"><i
                                                    class="bi bi-info-circle-fill text-secondary me-2"></i>Deskripsi
                                            </h5>
                                            <p class="text-muted" style="text-align: justify; line-height: 1.6;">
                                                {{ $item->deskripsi }}
                                            </p>
                                        </div>

                                        <!-- Skills -->
                                        <div>
                                            <h5 class="fw-semibold mb-2"><i
                                                    class="bi bi-info-circle-fill text-secondary me-2"></i>Skill
                                            </h5>
                                            <p class="text-muted" style="text-align: justify; line-height: 1.6;">
                                                {{ $item->skills }}
                                            </p>
                                        </div>
                                    </div>

                                    <!-- Footer -->
                                    <div class="modal-footer bg-light border-top-0 d-flex justify-content-end">
                                        <button class="btn btn-primary" data-bs-toggle="modal"
                                            data-bs-target="#daftarModal{{ $item->id }}">
                                            <i class="bi bi-send-fill me-1"></i> Lamar Sekarang
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </div>



                        {{-- Modal Daftar --}}
                        <div class="modal fade" id="daftarModal{{ $item->id }}" tabindex="-1"
                            aria-hidden="true">
                            <div class="modal-dialog modal-dialog-centered">
                                <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">

                                    <!-- Header -->
                                    <div class="bg-primary text-white p-4">
                                        <h5 class="modal-title mb-1">Lamar di {{ $item->nama_perusahaan }}</h5>
                                        <p class="mb-0 small">Posisi: <strong>{{ $item->posisi }}</strong></p>
                                    </div>

                                    <form id="formLamaran{{ $item->id }}"
                                        action="{{ route('alumni-siswa.loker.store', $item->id) }}" method="POST"
                                        enctype="multipart/form-data">
                                        @csrf

                                        <!-- Body -->
                                        <div class="modal-body p-4">
                                            <div class="mb-4">
                                                <label for="cv{{ $item->id }}"
                                                    class="form-label fw-semibold">Unggah CV Anda <span
                                                        class="text-danger">*</span></label>
                                                <input type="file" name="cv" id="cv{{ $item->id }}"
                                                    class="form-control" accept=".pdf" required>
                                                <div class="form-text">Hanya format PDF. Maksimal 2MB.</div>
                                            </div>
                                        </div>

                                        <!-- Footer -->
                                        <div
                                            class="modal-footer bg-light border-0 d-flex justify-content-between px-4 py-3">
                                            <button type="button" class="btn btn-outline-secondary"
                                                data-bs-dismiss="modal">Batal</button>

                                            <button type="button" class="btn btn-primary"
                                                onclick="confirmKirimLamaran('formLamaran{{ $item->id }}')">
                                                <i class="bi bi-send me-1"></i> Kirim Lamaran
                                            </button>
                                        </div>
                                    </form>


                                </div>
                            </div>
                        </div>
                    @endforeach

                    <div class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                        <div class="small text-muted">
                            Showing
                            <strong>{{ $loker->firstItem() ?? 0 }}</strong>
                            to
                            <strong>{{ $loker->lastItem() ?? 0 }}</strong>
                            of
                            <strong>{{ $loker->total() }}</strong>
                            entries
                        </div>

                        <nav>
                            <ul class="pagination mb-0">
                                {{-- Previous Page Link --}}
                                @if ($loker->onFirstPage())
                                    <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                                @else
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $loker->previousPageUrl() }}" rel="prev">&laquo;</a></li>
                                @endif

                                {{-- Pagination Elements --}}
                                @foreach ($loker->getUrlRange(1, $loker->lastPage()) as $page => $url)
                                    @if ($page == $loker->currentPage())
                                        <li class="page-item active"><span
                                                class="page-link">{{ $page }}</span>
                                        </li>
                                    @elseif ($page == 1 || $page == $loker->lastPage() || abs($page - $loker->currentPage()) <= 1)
                                        <li class="page-item"><a class="page-link"
                                                href="{{ $url }}">{{ $page }}</a></li>
                                    @elseif ($page == 2 || $page == $loker->lastPage() - 1)
                                        <li class="page-item disabled"><span class="page-link">...</span></li>
                                    @endif
                                @endforeach

                                {{-- Next Page Link --}}
                                @if ($loker->hasMorePages())
                                    <li class="page-item"><a class="page-link" href="{{ $loker->nextPageUrl() }}"
                                            rel="next">&raquo;</a>
                                    </li>
                                @else
                                    <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                                @endif
                            </ul>
                        </nav>
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

        </div>


        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
        <script>
            @if (session('success'))
                Swal.fire({
                    icon: 'success',
                    title: 'Berhasil!',
                    text: '{{ session('success') }}',
                    timer: 3000,
                    showConfirmButton: false
                });
            @endif

            @if (session('error'))
                Swal.fire({
                    icon: 'error',
                    title: 'Gagal!',
                    text: '{{ session('error') }}',
                    timer: 3000,
                    showConfirmButton: false
                });
            @endif

            @if (session('message'))
                Swal.fire({
                    icon: '{{ session('alert-type') == 'warning' ? 'warning' : 'info' }}',
                    title: '{{ ucfirst(session('alert-type') ?? 'Info') }}',
                    text: '{{ session('message') }}',
                    timer: 3000,
                    showConfirmButton: false
                });
            @endif
        </script>
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

            function confirmKirimLamaran(formId) {
                Swal.fire({
                    title: 'Kirim Lamaran?',
                    text: "Pastikan CV sudah benar sebelum dikirim.",
                    icon: 'question',
                    showCancelButton: true,
                    confirmButtonColor: '#0d6efd',
                    cancelButtonColor: '#6c757d',
                    confirmButtonText: 'Ya, kirim!',
                    cancelButtonText: 'Batal'
                }).then((result) => {
                    if (result.isConfirmed) {
                        const form = document.getElementById(formId);
                        if (form) {
                            form.submit();
                        } else {
                            console.error('Form tidak ditemukan:', formId);
                        }
                    }
                });
            }
        </script>
</body>

</html>
