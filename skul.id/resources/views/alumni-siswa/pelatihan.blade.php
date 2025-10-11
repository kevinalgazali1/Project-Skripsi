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
            background-color: #fff;
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
            margin: 0;
            overflow-x: hidden;
        }

        /* ============ Fitur Section ============ */
        #fitur {
            max-width: 1000px;
            margin: 2rem auto;
        }

        .banner {
            background: url('{{ url('img/background.jpg') }}') no-repeat center center;
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

            .banner-images img {
                width: 100%;
            }

            .banner .illustration {
                display: none;
            }

            .user-info {
                display: none;
                ;
            }
        }

        /* ============ Certificate Card Redesigned ============ */
        .certificate-grid {
            gap: 1rem;
            padding: 2rem;
            overflow-x: hidden;
        }

        .certificate-card {
            background: #fff;
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            max-width: 100%;
            height: 100%;
        }

        .certificate-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08);
        }

        .certificate-img {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-bottom: 1px solid #e2e8f0;
        }

        .certificate-content {
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }

        .certificate-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.5rem;
        }

        .certificate-provider,
        .certificate-date {
            font-size: 0.9rem;
            color: #6b7280;
            margin-bottom: 0.25rem;
        }

        .biaya {
            font-weight: 600;
            color: #111827;
            margin-bottom: 1rem;
        }

        .card-actions {
            display: flex;
            gap: 0.75rem;
        }

        .btn-view {
            padding: 0.5rem 1rem;
            font-size: 0.85rem;
            font-weight: 500;
            border-radius: 8px;
            border: none;
            background-color: #2563eb;
            color: white;
            transition: background-color 0.2s ease;
        }

        .btn-view:hover {
            background-color: #1d4ed8;
        }

        .btn-view.secondary {
            background-color: #f3f4f6;
            color: #374151;
        }

        .btn-view.secondary:hover {
            background-color: #e5e7eb;
        }

        /* Responsive Grid */
        @media (min-width: 768px) {
            .certificate-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 1rem;
            }
        }

        /* Container base */
        .container {
            max-width: 1140px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        /* Typography & Utility */
        .section-title {
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }

        .section-desc {
            font-size: 1rem;
            color: #6c757d;
            max-width: 700px;
            margin: 0 auto;
        }

        .text-primary {
            color: #007bff;
        }

        .text-success {
            color: #28a745;
        }

        .text-warning {
            color: #ffc107;
        }

        .text-white {
            color: #fff;
        }

        /* SECTION: Kenapa Sertifikasi Itu Penting */
        .section-highlight {
            background: url('{{ url('img/background.jpg') }}') no-repeat center center;
            padding: 4rem 1rem;
            width: 100%;
            min-height: 350px;
        }

        .section-highlight h2 {
            color: #c04055;
            font-weight: 700;
        }

        /* SECTION: Statistik */
        .section-stats {
            background: #ffffff;
            padding: 4rem 1rem;
        }

        .section-title {
            color: #c04055;
            font-weight: 700;
        }

        .stats-grid {
            display: flex;
            justify-content: space-around;
            flex-wrap: wrap;
            gap: 2rem;
        }

        .stat-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }

        .stat-number {
            font-size: 32px;
            font-weight: bold;
            margin: 5px 0;
        }

        .stat-item {
            flex: 1 1 200px;
        }

        .stat-number {
            font-size: 2rem;
            font-weight: 700;
        }

        .stat-label {
            font-size: 1rem;
            color: #6c757d;
        }

        /* SECTION: CTA */
        .section-cta {
            background: #ffffff;
            padding: 4rem 1rem;
            border-radius: 0.5rem;
        }

        .btn-cta {
            background-color: #fff;
            color: #007bff;
            padding: 0.75rem 1.5rem;
            font-weight: 600;
            text-decoration: none;
            border-radius: 0.3rem;
            display: inline-block;
            margin-top: 1rem;
        }

        .btn-cta:hover {
            background-color: #e2e6ea;
        }

        .partner-logos {
            position: absolute;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
            width: 100%;
        }

        .partner-logo {
            width: 100px;
            height: auto;
            display: block;
            margin: 0 auto;
        }

        .training-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            padding: 1.2rem;
        }

        @media (min-width: 1200px) {
            .training-grid {
                grid-template-columns: repeat(4, 1fr);
            }
        }

        .training-card {
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 16px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.04);
            transition: all 0.3s ease-in-out;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            cursor: pointer;
        }


        .training-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
        }

        .training-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #1f2937;
            margin-bottom: 0.75rem;
        }

        .training-info {
            font-size: 0.9rem;
            color: #4b5563;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .training-badge {
            display: inline-block;
            font-size: 0.75rem;
            padding: 0.35rem 0.8rem;
            border-radius: 999px;
            font-weight: 600;
            margin-top: 0.75rem;
            width: fit-content;
        }

        .badge-free {
            background-color: #dcfce7;
            color: #15803d;
        }

        .badge-paid {
            background-color: #fef3c7;
            color: #92400e;
        }

        /* Optional: Responsive adjustments */
        @media (max-width: 768px) {
            .training-grid {
                flex-direction: column;
                align-items: center;
            }

            .training-card {
                width: 90%;
            }
        }

        .training-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .btn-sm-primary {
            background-color: #2563eb;
            color: #fff;
            padding: 0.35rem 0.8rem;
            font-size: 0.75rem;
            border-radius: 6px;
            font-weight: 500;
            border: none;
            transition: background-color 0.2s ease;
        }

        .btn-sm-primary:hover {
            background-color: #1d4ed8;
        }

        .btn-sm-outline {
            background-color: #f3f4f6;
            color: #1f2937;
            padding: 0.35rem 0.8rem;
            font-size: 0.75rem;
            border-radius: 6px;
            border: none;
            font-weight: 500;
            transition: background-color 0.2s ease;
        }

        .btn-sm-outline:hover {
            background-color: #e5e7eb;
        }

        .modal-content {
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 1rem;
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
            <i class="bi bi-list fs-3" id="sideBarOff"></i>
        </button>
    </div>

    <!-- Offcanvas Sidebar -->
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

        <!-- Content -->
        <div class="content">
            <section class="banner position-relative overflow-hidden rounded-4 shadow-sm mb-5 text-dark"
                style="min-height: 500px;">
                <!-- Content -->
                <div
                    class="container h-100 position-relative z-2 d-flex flex-column justify-content-center align-items-center text-center p-4 p-md-5">
                    <h1 class="display-5 fw-bold mb-3 text-danger">Temukan Pelatihan Terbaik untuk Masa Depanmu</h1>
                    <p class="lead mb- text-muted">Ikuti pelatihan dari berbagai bidang dan tingkatkan keterampilan
                        sesuai
                        kebutuhan industri saat ini.</p>

                    <!-- Search Form -->
                    <form method="GET" action="{{ route('alumni-siswa.pelatihan') }}"
                        class="row g-2 align-items-center mb-4">

                        <div class="col-md-4">
                            <input type="text" name="nama" class="form-control"
                                placeholder="Cari nama pelatihan..." value="{{ request('nama') }}">
                        </div>

                        <div class="col-md-3">
                            <input type="text" name="kota" class="form-control" placeholder="Cari kota..."
                                value="{{ request('kota') }}">
                        </div>

                        <div class="col-md-3">
                            <select name="status" class="form-select">
                                <option value="">Semua Status</option>
                                <option value="Gratis" {{ request('status') == 'Gratis' ? 'selected' : '' }}>Gratis
                                </option>
                                <option value="Berbayar" {{ request('status') == 'Berbayar' ? 'selected' : '' }}>
                                    Berbayar</option>
                            </select>
                        </div>

                        <div class="col-md-2">
                            <button type="submit" class="btn btn-danger w-100">
                                <i class="bi bi-search"></i> Cari
                            </button>
                        </div>
                    </form>


                    <!-- Statistik Section -->
                    <div class="d-flex flex-wrap justify-content-center gap-4 mt-5">
                        <div class="text-center px-3">
                            <i class="bi bi-people-fill fs-2 text-info mb-2"></i>
                            <h2 class="fw-bold text-info mb-0 count-up" data-count="{{ $totalAlumni }}">
                                {{ $totalAlumni }}</h2>
                            <small class="text-muted">Peserta Terdaftar</small>
                        </div>
                        <div class="text-center px-3">
                            <i class="bi bi-easel2-fill fs-2 text-danger mb-2"></i>
                            <!-- Lebih representatif untuk pelatihan -->
                            <h2 class="fw-bold text-danger mb-0 count-up" data-count="{{ $totalPelatihan }}">
                                {{ $totalPelatihan }}</h2>
                            <small class="text-muted">Program Pelatihan</small>
                        </div>
                        <div class="text-center px-3">
                            <i class="bi bi-calendar-check-fill fs-2 text-info mb-2"></i> <!-- Terlaksana / sukses -->
                            <h2 class="fw-bold text-info mb-0 count-up" data-count="{{ $pelatihanSelesai }}">
                                {{ $pelatihanSelesai }}</h2>
                            <small class="text-muted">Pelatihan Terlaksana</small>
                        </div>
                    </div>

                </div>
            </section>


            @if ($pelatihan->isEmpty())
                <div class="d-flex justify-content-center align-items-center">
                    <div class="alert alert-warning text-center">
                        <i class="bi bi-exclamation-triangle-fill me-2"></i>
                        Tidak ada pelatihan ditemukan.
                    </div>
                </div>
            @else
                <div class="training-grid">
                    @foreach ($pelatihan as $item)
                        <div class="training-card" style="cursor: pointer;" data-bs-toggle="modal"
                            data-bs-target="#detailModal{{ $item->id }}">
                            <div>
                                <div class="training-title">{{ $item->nama_pelatihan }}</div>
                                <div class="training-info">
                                    <i class="bi bi-geo-alt-fill"></i> {{ $item->kota }}
                                </div>
                                <div class="training-info">
                                    <i class="bi bi-calendar3"></i>
                                    {{ \Carbon\Carbon::parse($item->tanggal_mulai)->translatedFormat('d M') }} -
                                    {{ \Carbon\Carbon::parse($item->tanggal_selesai)->translatedFormat('d M Y') }}
                                </div>
                                <div class="training-badge {{ $item->biaya == 0 ? 'badge-free' : 'badge-paid' }}">
                                    {{ $item->biaya == 0 ? 'Gratis' : 'Rp ' . number_format((float) str_replace('.', '', $item->biaya), 0, ',', '.') }}
                                </div>
                            </div>
                        </div>

                        <!-- Modal Detail -->
                        <div class="modal fade" id="detailModal{{ $item->id }}" tabindex="-1"
                            aria-labelledby="detailModalLabel{{ $item->id }}" aria-hidden="true">
                            <div class="modal-dialog modal-dialog-centered modal-xl">
                                <div class="modal-content border-0 rounded-4 overflow-hidden shadow-lg">
                                    <div class="row g-0">
                                        <!-- LEFT: Image Section -->
                                        <div class="col-md-5"
                                            style="background: url('{{ $item->foto_pelatihan ?? 'https://source.unsplash.com/600x800/?training,workshop' }}') center center / cover no-repeat;">
                                        </div>
                                        <!-- RIGHT: Content -->
                                        <div class="col-md-7 bg-white p-4 d-flex flex-column justify-content-between">
                                            <div>
                                                <div class="d-flex justify-content-between align-items-start mb-3">
                                                    <h4 class="fw-bold mb-0 text-dark"
                                                        id="detailModalLabel{{ $item->id }}">
                                                        {{ $item->nama_pelatihan }}
                                                    </h4>
                                                    <button type="button" class="btn-close"
                                                        data-bs-dismiss="modal"></button>
                                                </div>
                                                <ul class="list-unstyled mb-4">
                                                    <li class="mb-2 text-muted"><strong
                                                            class="text-dark">Kota:</strong>
                                                        {{ $item->kota }}</li>
                                                    <li class="mb-2 text-muted"><strong
                                                            class="text-dark">Tempat:</strong>
                                                        {{ $item->tempat ?? '-' }}</li>
                                                    <li class="mb-2 text-muted"><strong
                                                            class="text-dark">Waktu:</strong>
                                                        {{ \Carbon\Carbon::parse($item->tanggal_mulai)->translatedFormat('d M Y') }}
                                                        -
                                                        {{ \Carbon\Carbon::parse($item->tanggal_selesai)->translatedFormat('d M Y') }}
                                                    </li>
                                                    <li class="mb-2 text-muted"><strong
                                                            class="text-dark">Biaya:</strong>
                                                        {{ $item->biaya == 0 ? 'Gratis' : 'Rp ' . number_format((float) str_replace('.', '', $item->biaya), 0, ',', '.') }}
                                                    </li>
                                                </ul>
                                                <div class="mb-4" style="max-height: 150px; overflow-y: auto;">
                                                    <p class="text-muted"><strong
                                                            class="text-dark">Deskripsi:</strong><br>
                                                        {!! nl2br(e($item->deskripsi)) !!}
                                                    </p>
                                                </div>
                                            </div>
                                            <div class="text-end">
                                                <form id="formDaftarGratis{{ $item->id }}"
                                                    action="{{ route('alumni-siswa.pelatihan.store', $item->id) }}"
                                                    method="POST">
                                                    @csrf
                                                    <input type="hidden" name="pelatihan_id"
                                                        value="{{ $item->id }}">

                                                    @if (strtolower($item->status) === 'berbayar')
                                                        <button type="button" class="btn-view"
                                                            data-bs-toggle="modal"
                                                            data-bs-target="#modalBukti{{ $item->id }}">Daftar</button>
                                                    @else
                                                        <button type="button" class="btn-view"
                                                            onclick="confirmDaftarGratis('formDaftarGratis{{ $item->id }}', event)">
                                                            Daftar Gratis
                                                        </button>
                                                    @endif
                                                </form>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>

                        {{-- Modal Bukti --}}
                        <div class="modal fade" id="modalBukti{{ $item->id }}" tabindex="-1"
                            aria-labelledby="modalBuktiLabel{{ $item->id }}" aria-hidden="true">
                            <div class="modal-dialog">
                                <form id="formTransfer{{ $item->id }}"
                                    onsubmit="return confirmTransfer({{ $item->id }})"
                                    action="{{ route('alumni-siswa.pelatihan.store', $item->id) }}" method="POST"
                                    enctype="multipart/form-data">
                                    @csrf
                                    <input type="hidden" name="sertifikasi_id" value="{{ $item->id }}">

                                    <div class="modal-content">
                                        <div class="modal-header">
                                            <h5 class="modal-title" id="modalBuktiLabel{{ $item->id }}">Upload
                                                Bukti
                                                Transfer</h5>
                                            <button type="button" class="btn-close"
                                                data-bs-dismiss="modal"></button>
                                        </div>

                                        <div class="modal-body">
                                            <p>Silakan transfer ke rekening berikut:</p>
                                            <p><strong>{{ $item->nomor_rekening }}</strong></p>
                                            <div class="mb-3">
                                                <label for="bukti_{{ $item->id }}" class="form-label">Bukti
                                                    Transfer</label>
                                                <input type="file" name="bukti_transfer" class="form-control"
                                                    id="bukti_{{ $item->id }}" accept="image/*" required>
                                            </div>
                                        </div>

                                        <div class="modal-footer">
                                            <button type="button" class="btn btn-secondary"
                                                data-bs-dismiss="modal">Batal</button>
                                            <button type="submit" class="btn btn-primary">Kirim Bukti &
                                                Daftar</button>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </div>
                    @endforeach
                </div>
            @endif

            <div class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                <div class="small text-muted">
                    Showing
                    <strong>{{ $pelatihan->firstItem() ?? 0 }}</strong>
                    to
                    <strong>{{ $pelatihan->lastItem() ?? 0 }}</strong>
                    of
                    <strong>{{ $pelatihan->total() }}</strong>
                    entries
                </div>

                <nav>
                    <ul class="pagination mb-0">
                        {{-- Previous Page Link --}}
                        @if ($pelatihan->onFirstPage())
                            <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                        @else
                            <li class="page-item"><a class="page-link" href="{{ $pelatihan->previousPageUrl() }}"
                                    rel="prev">&laquo;</a>
                            </li>
                        @endif

                        {{-- Pagination Elements --}}
                        @foreach ($pelatihan->getUrlRange(1, $pelatihan->lastPage()) as $page => $url)
                            @if ($page == $pelatihan->currentPage())
                                <li class="page-item active"><span class="page-link">{{ $page }}</span>
                                </li>
                            @elseif ($page == 1 || $page == $pelatihan->lastPage() || abs($page - $pelatihan->currentPage()) <= 1)
                                <li class="page-item"><a class="page-link"
                                        href="{{ $url }}">{{ $page }}</a></li>
                            @elseif ($page == 2 || $page == $pelatihan->lastPage() - 1)
                                <li class="page-item disabled"><span class="page-link">...</span></li>
                            @endif
                        @endforeach

                        {{-- Next Page Link --}}
                        @if ($pelatihan->hasMorePages())
                            <li class="page-item"><a class="page-link" href="{{ $pelatihan->nextPageUrl() }}"
                                    rel="next">&raquo;</a>
                            </li>
                        @else
                            <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                        @endif
                    </ul>
                </nav>
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
    </div>


    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script src="https://cdn.jsdelivr.net/npm/countup.js@2.6.2/dist/countUp.umd.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {
            const CountUp = window.countUp.CountUp;

            const countUpElements = document.querySelectorAll('.count-up');
            const options = {
                threshold: 0.6
            };

            const observer = new IntersectionObserver((entries, observer) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const el = entry.target;
                        const target = parseInt(el.getAttribute('data-count'));

                        const countUp = new CountUp(el, target, {
                            duration: 2,
                            separator: '.'
                        });

                        if (!countUp.error) {
                            countUp.start();
                            observer.unobserve(el);
                        } else {
                            console.error(countUp.error);
                        }
                    }
                });
            }, options);

            countUpElements.forEach(el => observer.observe(el));
        });
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

        function confirmDaftarGratis(formId, event) {
            event.preventDefault();
            Swal.fire({
                title: 'Yakin ingin daftar?',
                text: "Setelah mendaftar, kamu tidak bisa membatalkannya.",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#3085d6',
                cancelButtonColor: '#aaa',
                confirmButtonText: 'Ya, daftar',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById(formId).submit();
                }
            });
        }

        function confirmTransfer(id) {
            return Swal.fire({
                title: 'Konfirmasi Pendaftaran',
                text: 'Pastikan bukti transfer sudah benar.',
                icon: 'info',
                showCancelButton: true,
                confirmButtonText: 'Ya, kirim!',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('formTransfer' + id).submit();
                }
            }), false; // Mencegah submit default
        }
    </script>
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

</body>

</html>
