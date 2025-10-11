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
        }

        body:not(.modal-open) {
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

        .main-content {
            width: 100%;
        }

        .right-section {
            background-color: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .sertif-item {
            padding: 10px;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
        }

        .sertif-container {
            padding: 2rem;
            border-radius: 15px;
            margin-top: 1.5rem;
        }

        .sertif-list .item {
            padding: 1rem;
            border-bottom: 1px solid #eee;
        }

        .sertif-list .item:last-child {
            border-bottom: none;
        }

        .sertif-list .item h6 {
            font-weight: 600;
            font-size: 15px;
        }

        .sertif-list .item p {
            font-size: 14px;
            margin: 0;
        }

        #konten {
            background-color: #fff;
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .navbar .logo {
            width: 80px;
            margin-left: 30px;
        }

        .search-bar {
            margin-bottom: 2rem;
        }

        .card-jurusan {
            background: #fff;
            border-radius: 15px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s ease;
            height: 100%;
        }

        .card-jurusan:hover {
            transform: scale(1.03);
        }

        .card-jurusan img {
            width: 60px;
            height: 60px;
            object-fit: contain;
            margin-bottom: 10px;
        }

        .card-jurusan h6 {
            font-size: 14px;
            font-weight: 600;
        }

        .right-section {
            background-color: white;
            border-radius: 15px;
            padding: 1.5rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        }

        .right-section h6 {
            font-weight: 600;
            margin-bottom: 1rem;
        }

        .sertif-item {
            padding: 10px;
            margin-bottom: 10px;
            border-bottom: 1px solid #eee;
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

        .content {
            flex: 1;
            overflow: auto;
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

        .modal-content {
            max-height: 90vh;
            overflow: hidden;
            animation: fadeInUp 0.3s ease-in-out;
        }

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

        /* WebKit (Chrome, Safari, Edge) */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }

        ::-webkit-scrollbar-track {
            background: transparent;
        }

        ::-webkit-scrollbar-thumb {
            /* Warna default semi-transparan */
            border-radius: 4px;
            border: none;
        }

        /* Hilangkan panah (scrollbar buttons) */
        ::-webkit-scrollbar-button {
            display: none;
        }

        /* Firefox */
        * {
            scrollbar-width: thin;
            scrollbar-color: rgba(0, 0, 0, 0.2) transparent;
        }

        #success-message,
        #error-message {
            animation: fadeSlideDown 0.5s ease;
            margin-bottom: 1.5rem;
        }

        @keyframes fadeSlideDown {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }

            to {
                opacity: 1;
                transform: translateY(0);
            }
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

            <div class="main-content">
                <div class="row">
                    <!-- Konten Utama -->
                    <div class="col-lg-12">
                        <div id="konten">
                            <div class="d-flex justify-content-between align-items-center mb-4">
                                <div>
                                    <h2 class="fw-bold mb-1">Kelola Pelatihan</h2>
                                    <p class="text-muted">Tambah dan kelola pelatihan atau sertifikasi dari mitra
                                        untuk alumni SMK.</p>
                                </div>
                            </div>

                            <hr class="container my-3">
                            <!-- Pelatihan Anda -->
                            <h5 class="mb-4 fw-bold">Pelatihan Anda</h5>

                            <form method="GET" action="" class="row g-2 mb-4 mx-4 gap-1 mt-2 align-items-end">
                                <div class="col-md-3">
                                    <label class="form-label fw-semibold small text-muted">Nama Pelatihan</label>
                                    <input type="text" class="form-control" name="judul"
                                        placeholder="Nama Pelatihan" value="{{ request('judul') }}">
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label fw-semibold small text-muted">Kota</label>
                                    <input type="text" class="form-control" name="kota" placeholder="Kota"
                                        value="{{ request('kota') }}">
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label fw-semibold small text-muted">Tahun Mulai</label>
                                    <input type="number" class="form-control" name="tahun_mulai"
                                        placeholder="Tahun Mulai" value="{{ request('tahun_mulai') }}">
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label fw-semibold small text-muted">Status Selesai</label>
                                    <select class="form-select" name="status_selesai">
                                        <option value="">Semua</option>
                                        <option value="selesai"
                                            {{ request('status_selesai') == 'selesai' ? 'selected' : '' }}>Selesai
                                        </option>
                                        <option value="belum"
                                            {{ request('status_selesai') == 'belum' ? 'selected' : '' }}>Belum
                                            Selesai</option>
                                    </select>
                                </div>
                                <div class="col-md-2">
                                    <label class="form-label fw-semibold small text-muted">Status</label>
                                    <select class="form-select" name="status">
                                        <option value="">Semua Status</option>
                                        <option value="Gratis" {{ request('status') == 'Gratis' ? 'selected' : '' }}>
                                            Gratis</option>
                                        <option value="Berbayar"
                                            {{ request('status') == 'Berbayar' ? 'selected' : '' }}>Berbayar
                                        </option>
                                    </select>
                                </div>
                                <div class="col-md-12 d-flex justify-content-end gap-2">
                                    <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                        <i class="bi bi-search me-1"></i> Cari
                                    </button>
                                    <a href="{{ route('mitra.pelatihan') }}"
                                        class="btn btn-outline-secondary rounded-pill px-4 shadow-sm">
                                        <i class="bi bi-arrow-clockwise me-1"></i> Reset
                                    </a>
                                    <a href="{{ route('mitra.pelatihan.export', request()->query()) }}"
                                        class="btn btn-outline-success rounded-pill px-4 shadow-sm">
                                        <i class="bi bi-file-earmark-excel me-1"></i> Excel
                                    </a>
                                    <button type="button" class="btn btn-success rounded-pill px-4 shadow-sm"
                                        data-bs-toggle="modal" data-bs-target="#pelatihanModal">
                                        <i class="bi bi-plus-circle me-1"></i> Tambah
                                    </button>
                                </div>
                            </form>

                            <!-- Pelatihan Card Grid -->
                            <div class="table-responsive">
                                <table class="table table-hover align-middle">
                                    <thead class="table-light text-center text-uppercase small">
                                        <tr>
                                            <th>No</th>
                                            <th>Foto</th>
                                            <th>Nama Pelatihan</th>
                                            <th>Kota</th>
                                            <th>Peserta</th>
                                            <th>Sekolah</th>
                                            <th>Jurusan</th>
                                            <th>Bekerja</th>
                                            <th>Tidak Bekerja</th>
                                            <th>Kuliah</th>
                                            <th>Aksi</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        @forelse($pelatihanMitraSendiri as $index => $pelatihan)
                                            <tr>
                                                <td class="text-center">{{ $index + 1 }}</td>
                                                <td class="text-center">
                                                    <img src="{{ asset('storage/' . $pelatihan->foto_pelatihan) }}"
                                                        alt="{{ $pelatihan->nama_pelatihan }}" width="100"
                                                        style="object-fit: cover; height: 70px;">
                                                </td>
                                                <td class="text-center">{{ $pelatihan->nama_pelatihan }}</td>
                                                <td class="text-center">{{ $pelatihan->kota }}</td>
                                                <td class="text-center">{{ $pelatihan->daftarPelatihan->count() }}
                                                </td>
                                                <td class="text-center">{{ $pelatihan->jumlah_asal_sekolah }}</td>
                                                <td class="text-center">{{ $pelatihan->jumlah_jurusan }}</td>
                                                <td class="text-center">{{ $pelatihan->jumlah_bekerja_dan_usaha }}
                                                </td>
                                                <td class="text-center">{{ $pelatihan->jumlah_tidak_bekerja }}</td>
                                                <td class="text-center">{{ $pelatihan->jumlah_kuliah }}</td>
                                                <td class="text-center">
                                                    <div class="dropdown">
                                                        <button class="btn btn-sm btn-primary dropdown-toggle"
                                                            type="button" data-bs-toggle="dropdown"
                                                            aria-expanded="false">
                                                            Aksi
                                                        </button>
                                                        <ul class="dropdown-menu dropdown-menu-end">
                                                            <li>
                                                                <button class="dropdown-item" data-bs-toggle="modal"
                                                                    data-bs-target="#detailModal{{ $pelatihan->id }}">
                                                                    <i class="bi bi-eye me-2"></i> Lihat
                                                                </button>
                                                            </li>
                                                            @php $sudahMulai = \Carbon\Carbon::parse($pelatihan->tanggal_mulai)->isPast(); @endphp
                                                            @if (!$sudahMulai)
                                                                <li>
                                                                    <button class="dropdown-item"
                                                                        data-bs-toggle="modal"
                                                                        data-bs-target="#edit{{ $pelatihan->id }}">
                                                                        <i class="bi bi-pencil me-2"></i> Edit
                                                                    </button>
                                                                </li>
                                                            @endif
                                                            <li>
                                                                <form id="deleteForm-{{ $pelatihan->id }}"
                                                                    action="{{ route('mitra.pelatihan.destroy', $pelatihan->id) }}"
                                                                    method="POST" class="d-none">
                                                                    @csrf
                                                                    @method('DELETE')
                                                                </form>
                                                                <button type="button"
                                                                    class="dropdown-item text-danger"
                                                                    onclick="confirmDelete('{{ $pelatihan->id }}')">
                                                                    <i class="bi bi-trash me-2"></i> Hapus
                                                                </button>
                                                            </li>
                                                        </ul>
                                                    </div>
                                                </td>
                                            </tr>
                                        @empty
                                            <tr>
                                                <td colspan="11" class="text-center text-muted py-4">Belum ada
                                                    sertifikasi yang Anda buat.</td>
                                            </tr>
                                        @endforelse
                                    </tbody>
                                </table>
                            </div>

                            <div
                                class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                                <div class="small text-muted">
                                    Showing
                                    <strong>{{ $pelatihanMitraSendiri->firstItem() ?? 0 }}</strong>
                                    to
                                    <strong>{{ $pelatihanMitraSendiri->lastItem() ?? 0 }}</strong>
                                    of
                                    <strong>{{ $pelatihanMitraSendiri->total() }}</strong>
                                    entries
                                </div>

                                <nav>
                                    <ul class="pagination mb-0">
                                        {{-- Previous Page Link --}}
                                        @if ($pelatihanMitraSendiri->onFirstPage())
                                            <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                                        @else
                                            <li class="page-item"><a class="page-link"
                                                    href="{{ $pelatihanMitraSendiri->previousPageUrl() }}"
                                                    rel="prev">&laquo;</a></li>
                                        @endif

                                        {{-- Pagination Elements --}}
                                        @foreach ($pelatihanMitraSendiri->getUrlRange(1, $pelatihanMitraSendiri->lastPage()) as $page => $url)
                                            @if ($page == $pelatihanMitraSendiri->currentPage())
                                                <li class="page-item active"><span
                                                        class="page-link">{{ $page }}</span>
                                                </li>
                                            @elseif ($page == 1 || $page == $pelatihanMitraSendiri->lastPage() || abs($page - $pelatihanMitraSendiri->currentPage()) <= 1)
                                                <li class="page-item"><a class="page-link"
                                                        href="{{ $url }}">{{ $page }}</a></li>
                                            @elseif ($page == 2 || $page == $pelatihanMitraSendiri->lastPage() - 1)
                                                <li class="page-item disabled"><span class="page-link">...</span></li>
                                            @endif
                                        @endforeach

                                        {{-- Next Page Link --}}
                                        @if ($pelatihanMitraSendiri->hasMorePages())
                                            <li class="page-item"><a class="page-link"
                                                    href="{{ $pelatihanMitraSendiri->nextPageUrl() }}" rel="next">&raquo;</a>
                                            </li>
                                        @else
                                            <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                                        @endif
                                    </ul>
                                </nav>
                            </div>

                            <!-- Separator -->
                            <hr class="container my-5">

                            <!-- Pelatihan Mitra Lain -->
                            <h5 class="mb-4 fw-bold">Pelatihan dari Mitra Lain</h5>
                            <div class="sertif-container mt-4">
                                <div class="row row-cols-1 row-cols-md-3 g-4">
                                    @forelse($pelatihanMitraLain as $pelatihan)
                                        <div class="col">
                                            <div class="card h-100 border-0 shadow-sm">
                                                <img src="{{ asset('storage/' . $pelatihan->foto_pelatihan) }}"
                                                    class="card-img-top object-fit-cover"
                                                    alt="Foto Pelatihan {{ $pelatihan->nama_pelatihan }}"
                                                    style="height: 200px; object-fit: cover;">
                                                <div class="card-body d-flex flex-column">
                                                    <h6 class="fw-semibold mb-1">{{ $pelatihan->nama_pelatihan }}</h6>
                                                    <ul class="list-unstyled text-muted small mb-3">
                                                        <li><i class="bi bi-calendar-event"></i>
                                                            {{ \Carbon\Carbon::parse($pelatihan->tanggal_mulai)->format('d M Y') }}
                                                            -
                                                            {{ \Carbon\Carbon::parse($pelatihan->tanggal_selesai)->format('d M Y') }}
                                                        </li>
                                                        <li><i class="bi bi-geo-alt"></i> {{ $pelatihan->kota }},
                                                            {{ $pelatihan->tempat_pelatihan }}
                                                        </li>
                                                        <li><i class="bi bi-cash-stack"></i>
                                                            {{ $pelatihan->biaya == 0 ? 'Gratis' : 'Rp' . number_format((int) $pelatihan->biaya, 0, ',', '.') }}
                                                        </li>
                                                    </ul>
                                                </div>
                                            </div>
                                        </div>
                                    @empty
                                        <p class="text-muted">Belum ada pelatihan dari mitra lain.</p>
                                    @endforelse
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Modal Detail Pelatihan -->
                    @foreach ($pelatihanMitraSendiri as $pelatihan)
                        <div class="modal fade" id="detailModal{{ $pelatihan->id }}" tabindex="-1"
                            aria-hidden="true">
                            <div class="modal-dialog modal-dialog-centered modal-xl">
                                <div class="modal-content shadow-lg rounded-4">
                                    <div class="modal-header bg-primary text-white">
                                        <h5 class="modal-title">Detail Pelatihan</h5>
                                    </div>

                                    <div class="modal-body px-4 py-4">
                                        <div class="row g-4">
                                            {{-- Kolom Kiri: Detail Pelatihan --}}
                                            <div class="col-lg-6">
                                                @if ($pelatihan->foto_pelatihan)
                                                    <div class="mb-3">
                                                        <img src="{{ asset('storage/' . $pelatihan->foto_pelatihan) }}"
                                                            alt="Foto Pelatihan"
                                                            class="img-fluid rounded-3 shadow-sm border w-100"
                                                            style="max-height: 280px; object-fit: cover;">
                                                    </div>
                                                @endif

                                                <h4 class="fw-bold mb-2">{{ $pelatihan->nama_pelatihan }}</h4>

                                                {{-- Scrollable Deskripsi --}}
                                                <div class="mb-3">
                                                    <div style="max-height: 180px; overflow-y: auto;">
                                                        <p class="mb-0" style="white-space: pre-line;">
                                                            {!! nl2br(e($pelatihan->deskripsi)) !!}
                                                        </p>
                                                    </div>
                                                </div>
                                            </div>

                                            {{-- Kolom Kanan: Daftar Peserta --}}
                                            <div class="col-lg-6">
                                                <div class="mb-2"><strong>📍 Lokasi:</strong>
                                                    {{ $pelatihan->tempat }}, {{ $pelatihan->kota }}</div>
                                                <div class="mb-2"><strong>📅 Tanggal:</strong>
                                                    {{ $pelatihan->tanggal_mulai }} s/d
                                                    {{ $pelatihan->tanggal_selesai }}
                                                </div>
                                                <div class="mb-2">
                                                    <strong>📌 Status:</strong>
                                                    <span
                                                        class="badge bg-{{ $pelatihan->status == 'Aktif' ? 'success' : 'secondary' }}">
                                                        {{ $pelatihan->status }}
                                                    </span>
                                                </div>
                                                <div class="mb-2">
                                                    <strong>💳 Harga Pelatihan:</strong>
                                                    @php
                                                        $rawBiaya = str_replace('.', '', $pelatihan->biaya);
                                                        $cleanBiaya = is_numeric($rawBiaya) ? (int) $rawBiaya : 0;
                                                    @endphp
                                                    {{ $cleanBiaya > 0 ? 'Rp ' . number_format($cleanBiaya, 0, ',', '.') : 'Gratis' }}
                                                </div>

                                                <div
                                                    class="d-flex justify-content-between align-items-center mb-3 mt-5">
                                                    <h5 class="fw-semibold mb-0">👥 Peserta Terdaftar</h5>
                                                    <button
                                                        class="btn btn-sm btn-outline-success d-flex align-items-center btnDownloadPeserta"
                                                        data-id="{{ $pelatihan->id }}">
                                                        <i class="bi bi-download me-1"></i> Download Peserta
                                                    </button>
                                                </div>

                                                @if ($pelatihan->daftarPelatihan && $pelatihan->daftarPelatihan->count())
                                                    <div class="table-responsive border rounded-3"
                                                        style="max-height: 350px; overflow: auto;">
                                                        <table
                                                            class="table table-sm table-bordered align-middle mb-0 small text-nowrap">
                                                            <thead class="table-light sticky-top">
                                                                <tr class="text-center">
                                                                    <th>No</th>
                                                                    <th>Nama</th>
                                                                    <th>Email</th>
                                                                    <th>No HP</th>
                                                                    <th>Bukti TF</th>
                                                                </tr>
                                                            </thead>
                                                            <tbody>
                                                                @foreach ($pelatihan->daftarPelatihan as $peserta)
                                                                    <tr class="text-center">
                                                                        <td>{{ $loop->iteration }}</td>
                                                                        <td class="text-truncate"
                                                                            style="max-width: 120px;">
                                                                            {{ $peserta->nama_lengkap }}</td>
                                                                        <td class="text-truncate"
                                                                            style="max-width: 150px;">
                                                                            {{ $peserta->email }}</td>
                                                                        <td>{{ $peserta->no_hp }}</td>
                                                                        <td class="text-center">
                                                                            @if ($peserta->bukti_transfer)
                                                                                <img src="{{ asset($peserta->bukti_transfer) }}"
                                                                                    alt="Bukti Transfer"
                                                                                    class="img-thumbnail"
                                                                                    style="max-height: 50px; cursor: pointer;"
                                                                                    data-bs-toggle="modal"
                                                                                    data-bs-target="#imageModal"
                                                                                    data-image="{{ asset($peserta->bukti_transfer) }}"
                                                                                    data-close-parent="#detailModal{{ $pelatihan->id }}"
                                                                                    onclick="this.classList.add('was-open')">
                                                                            @else
                                                                                <span class="text-muted">Tidak
                                                                                    ada</span>
                                                                            @endif
                                                                        </td>
                                                                    </tr>
                                                                @endforeach
                                                            </tbody>
                                                        </table>
                                                    </div>
                                                @else
                                                    <p class="text-muted mt-2">Belum ada peserta yang
                                                        mendaftar.</p>
                                                @endif
                                            </div>
                                        </div>
                                    </div>

                                    <div class="modal-footer justify-content-end bg-light py-3">
                                        <button type="button" class="btn btn-secondary"
                                            data-bs-dismiss="modal">Tutup</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    @endforeach

                    <!-- Modal Preview Gambar -->
                    <div class="modal fade" id="imageModal" tabindex="-1" aria-hidden="true">
                        <div class="modal-dialog modal-dialog-centered modal-lg">
                            <div class="modal-content border-0 shadow">
                                <div class="modal-header">
                                    <h5 class="modal-title">Pratinjau Bukti Transfer</h5>
                                    <button type="button" class="btn-close" data-bs-dismiss="modal"
                                        aria-label="Tutup"></button>
                                </div>
                                <div class="modal-body text-center">
                                    <img id="modalImagePreview" src="" alt="Preview"
                                        class="img-fluid rounded shadow" style="max-height: 600px;">
                                </div>
                            </div>
                        </div>
                    </div>



                    <!-- Modal Form Tambah Pelatihan -->
                    <div class="modal fade" id="pelatihanModal" tabindex="-1" aria-labelledby="pelatihanModalLabel"
                        aria-hidden="true">
                        <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                            <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                                <div class="row g-0">
                                    <!-- Ilustrasi / Informasi -->
                                    <div class="col-md-5 d-none d-md-block bg-light position-relative">
                                        <div
                                            class="h-100 d-flex flex-column justify-content-center align-items-center p-4">
                                            <h5 class="text-primary fw-bold">Tambah Pelatihan Baru</h5>
                                            <p class="text-muted text-center px-3">
                                                Publikasikan program pelatihan Anda agar alumni dapat mengembangkan
                                                keterampilan dan meningkatkan daya saing mereka.
                                            </p>
                                        </div>
                                    </div>
                                    <!-- Form -->
                                    <div class="col-md-7 bg-white">
                                        <form action="{{ route('mitra.pelatihan.store') }}" method="POST"
                                            enctype="multipart/form-data" class="p-4" id="tambahPelatihanForm">
                                            @csrf
                                            <div class="modal-header border-0 pb-0">
                                                <h5 class="modal-title fw-bold" id="pelatihanModalLabel">
                                                    <i class="bi bi-journal-text me-2 text-primary"></i>Tambah
                                                    Pelatihan
                                                </h5>
                                            </div>
                                            <div class="modal-body overflow-auto" style="max-height: 65vh;">
                                                <div class="mb-3">
                                                    <label for="nama_pelatihan" class="form-label">Judul
                                                        Pelatihan</label>
                                                    <div class="input-group">
                                                        <span class="input-group-text"><i
                                                                class="bi bi-card-text"></i></span>
                                                        <input type="text" class="form-control"
                                                            id="nama_pelatihan" name="nama_pelatihan"
                                                            placeholder="Contoh: Pelatihan UI/UX Designer" required>
                                                    </div>
                                                </div>
                                                <div class="mb-3">
                                                    <label for="deskripsi" class="form-label">Deskripsi</label>
                                                    <textarea class="form-control" id="deskripsi" name="deskripsi" rows="3"
                                                        placeholder="Jelaskan pelatihan ini secara singkat..." required></textarea>
                                                </div>
                                                <div class="row">
                                                    <div class="col-md-6 mb-3">
                                                        <label for="tanggal_mulai" class="form-label">Waktu
                                                            Mulai</label>
                                                        <input type="date" class="form-control" id="tanggal_mulai"
                                                            name="tanggal_mulai" required>
                                                    </div>
                                                    <div class="col-md-6 mb-3">
                                                        <label for="tanggal_selesai" class="form-label">Waktu
                                                            Selesai</label>
                                                        <input type="date" class="form-control"
                                                            id="tanggal_selesai" name="tanggal_selesai" required>
                                                    </div>
                                                </div>
                                                <div class="row">
                                                    <div class="col-md-6 mb-3">
                                                        <label for="kota" class="form-label">Kota
                                                            Penyelenggaraan</label>
                                                        <input type="text" class="form-control" id="kota"
                                                            name="kota" placeholder="Contoh: Bandung" required>
                                                    </div>
                                                    <div class="col-md-6 mb-3">
                                                        <label for="tempat_pelatihan" class="form-label">Tempat
                                                            Penyelenggaraan</label>
                                                        <input type="text" class="form-control"
                                                            id="tempat_pelatihan" name="tempat_pelatihan"
                                                            placeholder="Contoh: Gedung Serbaguna" required>
                                                    </div>
                                                </div>
                                                <div class="mb-3">
                                                    <label for="status" class="form-label">Status
                                                        Pelatihan</label>
                                                    <select class="form-select" id="status" name="status"
                                                        required onchange="toggleBiaya()">
                                                        <option value="Berbayar">Berbayar</option>
                                                        <option value="Gratis" selected>Gratis</option>
                                                    </select>
                                                </div>
                                                <div class="mb-3" id="biayaGroup">
                                                    <label for="biaya" class="form-label">Biaya Pelatihan
                                                        (Rp)</label>
                                                    <input type="string" class="form-control" id="biaya"
                                                        name="biaya" placeholder="Contoh: 250000"
                                                        oninput="formatRupiah(this)" required>
                                                </div>
                                                <div class="mb-3" id="rekeningGroup" style="display: none;">
                                                    <label for="nomor_rekening" class="form-label">Nomor
                                                        Rekening</label>
                                                    <input type="text" class="form-control" id="nomor_rekening"
                                                        name="nomor_rekening"
                                                        placeholder="Contoh: 1234567890 a.n PT Pelatihan BCA">
                                                </div>
                                                <div class="mb-3">
                                                    <label for="foto_pelatihan" class="form-label">Upload Foto
                                                        Pelatihan</label>
                                                    <input type="file" class="form-control" id="foto_pelatihan"
                                                        name="foto_pelatihan" accept=".jpg,.jpeg,.png" required>
                                                </div>
                                                <div class="modal-footer border-0 pt-0">
                                                    <button type="button" class="btn btn-outline-secondary"
                                                        data-bs-dismiss="modal">Batal</button>
                                                    <button type="button" class="btn btn-primary"
                                                        onclick="confirmTambahPelatihan()">Simpan
                                                        Pelatihan</button>
                                                </div>
                                            </div>
                                        </form>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    {{-- Modal Edit --}}
                    @foreach ($pelatihanMitraSendiri as $pelatihan)
                        <div class="modal fade" id="edit{{ $pelatihan->id }}" tabindex="-1"
                            aria-labelledby="editPelatihanModalLabel{{ $pelatihan->id }}" aria-hidden="true">
                            <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                                <div class="modal-content edit-modal-content">
                                    <form id="editForm{{ $pelatihan->id }}"
                                        action="{{ url('/mitra/update-pelatihan/' . $pelatihan->id) }}"
                                        method="POST" enctype="multipart/form-data">
                                        @csrf
                                        @method('PUT')
                                        <input type="hidden" name="action" value="update">
                                        <input type="hidden" name="id" value="{{ $pelatihan->id }}">

                                        <div class="modal-header border-0">
                                            <h5 class="modal-title fw-semibold text-primary-emphasis">
                                                <i class="bi bi-pencil-square me-2"></i> Edit Pelatihan
                                            </h5>
                                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                                aria-label="Close"></button>
                                        </div>

                                        <div class="modal-body">
                                            <div class="row gx-4 gy-3">
                                                <!-- Left Column -->
                                                <div class="col-lg-6">
                                                    <div class="form-group mb-3">
                                                        <label class="form-label">Judul Pelatihan</label>
                                                        <input type="text" class="form-control edit-input"
                                                            name="nama_pelatihan"
                                                            value="{{ $pelatihan->nama_pelatihan }}" required>
                                                    </div>

                                                    <div class="form-group mb-3">
                                                        <label class="form-label">Deskripsi</label>
                                                        <textarea class="form-control edit-input" name="deskripsi" rows="4" required>{{ $pelatihan->deskripsi }}</textarea>
                                                    </div>

                                                    <div class="row">
                                                        <div class="col-md-6 mb-3">
                                                            <label class="form-label">Tanggal Mulai</label>
                                                            <input type="date" class="form-control edit-input"
                                                                name="tanggal_mulai"
                                                                value="{{ $pelatihan->tanggal_mulai }}" required>
                                                        </div>
                                                        <div class="col-md-6 mb-3">
                                                            <label class="form-label">Tanggal Selesai</label>
                                                            <input type="date" class="form-control edit-input"
                                                                name="tanggal_selesai"
                                                                value="{{ $pelatihan->tanggal_selesai }}" required>
                                                        </div>
                                                    </div>
                                                </div>

                                                <!-- Right Column -->
                                                <div class="col-lg-6">
                                                    <div class="form-group mb-3">
                                                        <label class="form-label">Kota</label>
                                                        <input type="text" class="form-control edit-input"
                                                            name="kota" value="{{ $pelatihan->kota }}" required>
                                                    </div>

                                                    <div class="form-group mb-3">
                                                        <label class="form-label">Tempat</label>
                                                        <input type="text" class="form-control edit-input"
                                                            name="tempat_pelatihan"
                                                            value="{{ $pelatihan->tempat_pelatihan }}" required>
                                                    </div>

                                                    <div class="form-group mb-3">
                                                        <label class="form-label">Status Pelatihan</label>
                                                        <select class="form-select edit-input" name="status"
                                                            onchange="toggleBiayaField({{ $pelatihan->id }})"
                                                            required>
                                                            <option value="Berbayar"
                                                                {{ $pelatihan->status == 'Berbayar' ? 'selected' : '' }}>
                                                                Berbayar
                                                            </option>
                                                            <option value="Gratis"
                                                                {{ $pelatihan->status == 'Gratis' ? 'selected' : '' }}>
                                                                Gratis</option>
                                                        </select>
                                                    </div>

                                                    <div class="form-group mb-3"
                                                        id="edit_biaya_wrapper{{ $pelatihan->id }}">
                                                        <label class="form-label">Biaya (Rp)</label>
                                                        <input type="string" class="form-control edit-input"
                                                            name="biaya" oninput="formatRupiah(this)"
                                                            value="{{ $pelatihan->biaya }}">
                                                    </div>

                                                    <div class="form-group mb-3"
                                                        id="edit_rekening_wrapper{{ $pelatihan->id }}">
                                                        <label class="form-label">Nomor Rekening</label>
                                                        <input type="text" class="form-control edit-input"
                                                            name="nomor_rekening"
                                                            value="{{ $pelatihan->nomor_rekening }}">
                                                    </div>

                                                    <div class="form-group">
                                                        <label class="form-label">Upload Foto Pelatihan
                                                            (Opsional)
                                                        </label>
                                                        <input type="file" class="form-control edit-input"
                                                            name="foto_pelatihan" accept=".jpg,.jpeg,.png">
                                                    </div>
                                                </div>
                                            </div>
                                        </div>

                                        <div class="modal-footer border-top-0">
                                            <button type="button" class="btn btn-light"
                                                data-bs-dismiss="modal">Batal</button>
                                            <button type="button" class="btn btn-primary"
                                                onclick="confirmEditPelatihan({{ $pelatihan->id }})">Perbarui</button>
                                        </div>
                                    </form>
                                </div>
                            </div>
                        </div>
                    @endforeach

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
                                    mitraskul.Id adalah platform yang menghubungkan alumni dengan sekolah, dunia
                                    industri,
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
                                    <img src="{{ url('img/pu.png') }}" alt="PUP Makassar"
                                        class="partner-footer-logo">
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

                function confirmTambahPelatihan() {
                    Swal.fire({
                        title: 'Simpan Pelatihan?',
                        text: "Pastikan semua informasi sudah lengkap dan benar.",
                        icon: 'question',
                        showCancelButton: true,
                        confirmButtonColor: '#198754',
                        cancelButtonColor: '#6c757d',
                        confirmButtonText: 'Ya, Simpan',
                        cancelButtonText: 'Batal'
                    }).then((result) => {
                        if (result.isConfirmed) {
                            document.getElementById('tambahPelatihanForm').submit();
                        }
                    });
                }

                function confirmDelete(id) {
                    Swal.fire({
                        title: 'Yakin ingin menghapus?',
                        text: "Data pelatihan akan dihapus permanen!",
                        icon: 'warning',
                        showCancelButton: true,
                        confirmButtonColor: '#d33',
                        cancelButtonColor: '#6c757d',
                        confirmButtonText: 'Ya, hapus!',
                        cancelButtonText: 'Batal'
                    }).then((result) => {
                        if (result.isConfirmed) {
                            document.getElementById('deleteForm-' + id).submit();
                        }
                    });
                }

                function confirmEditPelatihan(id) {
                    Swal.fire({
                        title: 'Perbarui Pelatihan?',
                        text: "Pastikan semua data sudah benar.",
                        icon: 'question',
                        showCancelButton: true,
                        confirmButtonColor: '#0d6efd',
                        cancelButtonColor: '#6c757d',
                        confirmButtonText: 'Ya, perbarui!',
                        cancelButtonText: 'Batal'
                    }).then((result) => {
                        if (result.isConfirmed) {
                            document.getElementById('editForm' + id).submit();
                        }
                    });
                }

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
                document.addEventListener('DOMContentLoaded', function() {
                    const downloadButtons = document.querySelectorAll('.btnDownloadPeserta');

                    downloadButtons.forEach(button => {
                        button.addEventListener('click', function() {
                            const sertifikasiId = this.getAttribute('data-id');
                            window.location.href =
                                `/public/mitra/pelatihan/${sertifikasiId}/peserta/export`;
                        });
                    });
                });
            </script>
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const imageModal = new bootstrap.Modal(document.getElementById('imageModal'));
                    const modalImage = document.getElementById('modalImagePreview');

                    // Tangani klik semua gambar dengan target imageModal
                    document.querySelectorAll('[data-bs-target="#imageModal"]').forEach(img => {
                        img.addEventListener('click', function() {
                            const imageUrl = img.getAttribute('data-image');
                            const parentSelector = img.getAttribute('data-close-parent');
                            modalImage.src = imageUrl;

                            // Sembunyikan modal induk jika ada
                            if (parentSelector) {
                                const parentModalEl = document.querySelector(parentSelector);
                                if (parentModalEl) {
                                    const parentInstance = bootstrap.Modal.getInstance(parentModalEl);
                                    if (parentInstance) {
                                        parentInstance.hide();
                                        setTimeout(() => {
                                            imageModal.show();
                                        }, 300);
                                    }
                                }
                            } else {
                                imageModal.show();
                            }
                        });
                    });

                    // Tampilkan kembali modal sebelumnya setelah image modal ditutup
                    document.getElementById('imageModal').addEventListener('hidden.bs.modal', function() {
                        const openedFrom = document.querySelector('[data-close-parent].was-open');
                        if (openedFrom) {
                            const selector = openedFrom.getAttribute('data-close-parent');
                            const modalEl = document.querySelector(selector);
                            if (modalEl) {
                                const instance = bootstrap.Modal.getOrCreateInstance(modalEl);
                                instance.show();
                            }
                            openedFrom.classList.remove('was-open');
                        }
                    });
                });
            </script>
            <script>
                function formatRupiah(input) {
                    let value = input.value.replace(/\D/g, '');
                    input.value = new Intl.NumberFormat('id-ID').format(value);
                }

                function formatRupiah(input) {
                    let value = input.value.replace(/\D/g, ''); // Hapus semua karakter non-digit
                    let formatted = value.replace(/\B(?=(\d{3})+(?!\d))/g, '.'); // Tambah titik setiap 3 digit dari belakang
                    input.value = formatted;
                }

                function toggleBiayaField(id) {
                    const statusSelect = document.querySelector(`[name="status"][onchange*="${id}"]`);
                    const biayaWrapper = document.getElementById(`edit_biaya_wrapper${id}`);
                    const rekeningWrapper = document.getElementById(`edit_rekening_wrapper${id}`);
                    const biayaInput = biayaWrapper?.querySelector('input[name="biaya"]');
                    const rekeningInput = rekeningWrapper?.querySelector('input[name="nomor_rekening"]');

                    if (statusSelect) {
                        if (statusSelect.value === 'Gratis') {
                            if (biayaWrapper) biayaWrapper.style.display = 'none';
                            if (rekeningWrapper) rekeningWrapper.style.display = 'none';
                            if (biayaInput) biayaInput.value = 0;
                            if (rekeningInput) rekeningInput.value = '';
                        } else {
                            if (biayaWrapper) biayaWrapper.style.display = 'block';
                            if (rekeningWrapper) rekeningWrapper.style.display = 'block';
                        }
                    }
                }

                document.addEventListener('DOMContentLoaded', function() {
                    @foreach ($pelatihanMitraSendiri as $pelatihan)
                        toggleBiayaField({{ $pelatihan->id }});
                    @endforeach
                });

                function toggleBiaya() {
                    const status = document.getElementById('status').value;
                    const biayaWrapper = document.getElementById('biayaGroup');
                    const rekeningWrapper = document.getElementById('rekeningGroup');
                    const biayaInput = document.getElementById('biaya');

                    if (status === 'Gratis') {
                        biayaWrapper.style.display = 'none';
                        rekeningWrapper.style.display = 'none';
                        biayaInput.value = 0;
                    } else {
                        biayaWrapper.style.display = 'block';
                        rekeningWrapper.style.display = 'block';
                    }
                }

                // Panggil saat halaman selesai dimuat
                document.addEventListener('DOMContentLoaded', function() {
                    toggleBiaya();
                });

                const tanggalMulai = document.getElementById('tanggal_mulai');
                const tanggalSelesai = document.getElementById('tanggal_selesai');

                // Batasi tanggal mulai minimal hari ini
                const today = new Date().toISOString().split('T')[0];
                tanggalMulai.setAttribute('min', today);

                // Saat tanggal mulai berubah
                tanggalMulai.addEventListener('change', function() {
                    tanggalSelesai.value = ''; // kosongkan dulu agar tidak tertinggal nilai lama
                    tanggalSelesai.setAttribute('min', this.value); // tanggal selesai minimal = tanggal mulai
                });

                // Opsional: validasi tambahan saat submit (jika dibutuhkan)
                document.querySelector('form')?.addEventListener('submit', function(e) {
                    if (tanggalSelesai.value < tanggalMulai.value) {
                        alert('Tanggal selesai tidak boleh sebelum tanggal mulai.');
                        e.preventDefault();
                    }
                });
            </script>
</body>

</html>
