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

        .main-content {
            width: 100%;
        }

        .right-section {
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
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
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

        .card.loker-card {
            transition: all 0.3s ease;
            border-radius: 1rem;
            overflow: hidden;
            background-color: #fff;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        }

        .card.loker-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
        }

        .card.loker-card img {
            transition: transform 0.3s ease;
        }

        .card.loker-card:hover img {
            transform: scale(1.03);
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

        .modal-content {
            max-height: 90vh;
            overflow: hidden;
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
                            <div class="d-flex justify-content-between align-items-center mb-2">
                                <div>
                                    <h2 class="fw-bold mb-1">Kelola Lowongan Kerja</h2>
                                    <p class="text-muted">Tambah dan kelola lowongan kerja untuk alumni dari mitra Anda.
                                    </p>
                                </div>
                            </div>

                            <hr class="container my-3">

                            <h5 class="mb-4 fw-bold">Lowongan Kerja Anda</h5>
                            <form method="GET" action="" class="mb-4 px-4">
                                <div class="row g-3 mb-3">

                                    {{-- Nama Mitra --}}
                                    <div class="col-md-3">
                                        <label class="form-label fw-semibold small text-muted">Nama Perusahaan</label>
                                        <input type="text"
                                            class="form-control rounded-3 shadow-sm border border-light-subtle"
                                            name="nama_perusahaan" value="{{ request('nama_perusahaan') }}"
                                            placeholder="Contoh: PT Mitra Skul">
                                    </div>

                                    {{-- Lokasi --}}
                                    <div class="col-md-3">
                                        <label class="form-label fw-semibold small text-muted">Lokasi</label>
                                        <input type="text"
                                            class="form-control rounded-3 shadow-sm border border-light-subtle"
                                            name="lokasi" value="{{ request('lokasi') }}"
                                            placeholder="Contoh: Makassar">
                                    </div>

                                    {{-- Tipe --}}
                                    <div class="col-md-3">
                                        <label class="form-label fw-semibold small text-muted">Tipe Kerja</label>
                                        <select class="form-select" name="tipe">
                                            <option value="">Semua Tipe</option>
                                            <option value="Full Time"
                                                {{ request('tipe') == 'Full Time' ? 'selected' : '' }}>Full
                                                Time</option>
                                            <option value="Part Time"
                                                {{ request('tipe') == 'Part Time' ? 'selected' : '' }}>Part
                                                Time</option>
                                            <option value="Remote"
                                                {{ request('tipe') == 'Remote' ? 'selected' : '' }}>Remote
                                            </option>
                                        </select>
                                    </div>

                                    {{-- Status Lowongan --}}
                                    <div class="col-md-3">
                                        <label class="form-label fw-semibold small text-muted">Status Lowongan</label>
                                        <select class="form-select" name="status">
                                            <option value="">Semua Status</option>
                                            <option value="buka"
                                                {{ request('status') == 'buka' ? 'selected' : '' }}>Buka
                                            </option>
                                            <option value="tutup"
                                                {{ request('status') == 'tutup' ? 'selected' : '' }}>Tutup
                                            </option>
                                        </select>
                                    </div>

                                    {{-- Tombol --}}
                                    <div class="col d-flex justify-content-end gap-2 align-items-end">
                                        <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                            <i class="bi bi-search me-1"></i> Terapkan
                                        </button>
                                        <a href="{{ route('mitra.loker') }}"
                                            class="btn btn-outline-secondary rounded-pill px-4 shadow-sm">
                                            <i class="bi bi-arrow-clockwise me-1"></i> Reset
                                        </a>
                                        <a href="{{ route('mitra.loker.export', request()->query()) }}"
                                            class="btn btn-success rounded-pill px-4 shadow-sm">
                                            <i class="bi bi-file-earmark-excel me-1"></i> Excel
                                        </a>
                                        <button type="button"
                                            class="btn btn-outline-primary rounded-pill px-4 shadow-sm"
                                            data-bs-toggle="modal" data-bs-target="#modalTambah">
                                            <i class="bi bi-plus-circle me-1"></i> Tambah
                                        </button>
                                    </div>
                                </div>
                            </form>

                            {{-- Loker Mitra Sendiri --}}
                            <div class="table-responsive loker-table-wrapper">
                                <table class="table pelatihan-table table-hover align-middle mb-0"
                                    id="lokerMitraTable">
                                    <thead class="table-light text-center text-uppercase small">
                                        <tr>
                                            <th>No</th>
                                            <th>Foto</th>
                                            <th>Perusahaan</th>
                                            <th>Posisi</th>
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
                                        @forelse ($lokerMitraSendiri as $index => $l)
                                            <tr class="text-center">
                                                <td>{{ $index + 1 }}</td>
                                                <td><img src="{{ asset('storage/' . $l->gambar) }}"
                                                        alt="{{ $l->nama_perusahaan }}" width="100"
                                                        style="object-fit: cover; height: 70px;"></td>
                                                <td>{{ $l->nama_perusahaan }}</td>
                                                <td>{{ $l->posisi }}</td>
                                                <td>{{ $l->daftarLoker->count() }}</td>
                                                <td>{{ $l->jumlah_asal_sekolah ?? 0 }}</td>
                                                <td>{{ $l->jumlah_jurusan ?? 0 }}</td>
                                                <td>{{ $l->jumlah_bekerja_dan_usaha ?? 0 }}</td>
                                                <td>{{ $l->jumlah_tidak_bekerja ?? 0 }}</td>
                                                <td>{{ $l->jumlah_kuliah ?? 0 }}</td>
                                                <td>
                                                    <div class="dropdown">
                                                        <button class="btn btn-sm btn-primary dropdown-toggle"
                                                            type="button" data-bs-toggle="dropdown"
                                                            aria-expanded="false">
                                                            <i class="bi bi-three-dots-vertical"></i> Aksi
                                                        </button>
                                                        <ul class="dropdown-menu dropdown-menu-end shadow-sm"
                                                            data-bs-auto-close="outside">
                                                            <li>
                                                                <button
                                                                    class="dropdown-item d-flex align-items-center gap-2"
                                                                    data-bs-toggle="modal"
                                                                    data-bs-target="#modalLihat{{ $l->id }}">
                                                                    <i class="bi bi-eye text-secondary"></i> Lihat
                                                                </button>
                                                            </li>
                                                            <li>
                                                                <button
                                                                    class="dropdown-item d-flex align-items-center gap-2"
                                                                    data-bs-toggle="modal"
                                                                    data-bs-target="#editLokerModal"
                                                                    onclick="openEditLokerModal({
        id: '{{ $l->id }}',
        nama_perusahaan: '{{ $l->nama_perusahaan }}',
        posisi: '{{ $l->posisi }}',
        lokasi: '{{ $l->lokasi }}',
        tipe: '{{ $l->tipe }}',
        pendidikan: '{{ $l->pendidikan }}',
        deskripsi: `{!! $l->deskripsi !!}`,
        gaji: '{{ $l->gaji }}',
        status: '{{ $l->status }}',
        skills: '{{ $l->skills }}',
        foto: '{{ asset('storage/' . $l->gambar) }}'
    })">
                                                                    <i class="bi bi-pencil-square text-warning"></i>
                                                                    Edit
                                                                </button>
                                                            </li>
                                                            <li>
                                                                <form id="deleteForm{{ $l->id }}"
                                                                    action="{{ route('mitra.loker.destroy', $l->id) }}"
                                                                    method="POST" class="d-none">
                                                                    @csrf
                                                                    @method('DELETE')
                                                                </form>

                                                                <button type="button"
                                                                    onclick="confirmDelete({{ $l->id }})"
                                                                    class="dropdown-item text-danger d-flex align-items-center gap-2">
                                                                    <i class="bi bi-trash3-fill"></i> Hapus
                                                                </button>
                                                            </li>
                                                        </ul>
                                                    </div>
                                                </td>
                                            </tr>
                                        @empty
                                            <tr>
                                                <td colspan="11" class="text-muted text-center">Belum ada lowongan
                                                    kerja yang Anda buat.</td>
                                            </tr>
                                        @endforelse
                                    </tbody>
                                </table>
                            </div>

                            @foreach ($lokerMitraSendiri as $l)
                                <div class="modal fade" id="modalLihat{{ $l->id }}" tabindex="-1"
                                    aria-hidden="true">
                                    <div class="modal-dialog modal-dialog-centered modal-xl">
                                        <div class="modal-content shadow-lg rounded-4">
                                            <div class="modal-header bg-primary text-white">
                                                <h5 class="modal-title">Detail Lowongan Kerja</h5>
                                            </div>
                                            <div class="modal-body px-4 py-4">
                                                <div class="row g-4">
                                                    {{-- Kolom Kiri: Detail Loker --}}
                                                    <div class="col-lg-6">
                                                        @if ($l->gambar)
                                                            <div class="mb-3">
                                                                <img src="{{ asset('storage/' . $l->gambar) }}"
                                                                    alt="Foto Perusahaan"
                                                                    class="img-fluid rounded-3 shadow-sm border w-100"
                                                                    style="max-height: 280px; object-fit: cover;">
                                                            </div>
                                                        @endif

                                                        <h4 class="fw-bold mb-2">{{ $l->posisi }}</h4>

                                                        {{-- Deskripsi Scrollable --}}
                                                        <div class="mb-3">
                                                            <div style="max-height: 180px; overflow-y: auto;">
                                                                <p class="mb-0" style="white-space: pre-line;">
                                                                    {!! nl2br(e($l->deskripsi)) !!}
                                                                </p>
                                                            </div>
                                                        </div>
                                                    </div>

                                                    {{-- Kolom Kanan: Daftar Pelamar --}}
                                                    <div class="col-lg-6">
                                                        <div class="mb-2"><strong>🏢 Perusahaan:</strong>
                                                            {{ $l->nama_perusahaan }}</div>
                                                        <div class="mb-2"><strong>📍 Lokasi:</strong>
                                                            {{ $l->lokasi }}</div>
                                                        <div class="mb-2"><strong>🧑‍🎓
                                                                Pendidikan:</strong>
                                                            {{ $l->pendidikan }}</div>
                                                        <div class="mb-2"><strong>💼 Tipe:</strong>
                                                            {{ $l->tipe }}</div>
                                                        <div class="mb-2"><strong>💰 Gaji:</strong>
                                                            {{ $l->gaji }}</div>

                                                        <div
                                                            class="d-flex justify-content-between align-items-center mb-3 mt-5">
                                                            <h5 class="fw-semibold mb-0">👥 Pelamar
                                                                Terdaftar</h5>
                                                            {{-- Tombol Download CV (opsional) --}}
                                                            <button
                                                                class="btn btn-sm btn-outline-success btnDownloadPeserta"
                                                                data-id="{{ $l->id }}">>
                                                                <i class="bi bi-download me-1"></i> Export Data
                                                            </button>
                                                        </div>

                                                        @if ($l->daftarLoker && $l->daftarLoker->count())
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
                                                                            <th>CV</th>
                                                                        </tr>
                                                                    </thead>
                                                                    <tbody>
                                                                        @foreach ($l->daftarLoker as $pelamar)
                                                                            <tr class="text-center">
                                                                                <td>{{ $loop->iteration }}
                                                                                </td>
                                                                                <td class="text-truncate"
                                                                                    style="max-width: 120px;">
                                                                                    {{ $pelamar->nama_lengkap }}
                                                                                </td>
                                                                                <td class="text-truncate"
                                                                                    style="max-width: 150px;">
                                                                                    {{ $pelamar->email }}
                                                                                </td>
                                                                                <td>{{ $pelamar->no_hp }}
                                                                                </td>
                                                                                <td>
                                                                                    @if ($pelamar->cv)
                                                                                        <a href="{{ asset('storage/assets/cv/' . $pelamar->cv) }}"
                                                                                            target="_blank"
                                                                                            class="btn btn-sm btn-outline-info">
                                                                                            <i
                                                                                                class="bi bi-file-earmark-person"></i>
                                                                                            Lihat CV
                                                                                        </a>
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
                                                            <p class="text-muted mt-2">Belum ada pelamar
                                                                yang
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

                            <div
                                class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                                <div class="small text-muted">
                                    Showing
                                    <strong>{{ $lokerMitraSendiri->firstItem() ?? 0 }}</strong>
                                    to
                                    <strong>{{ $lokerMitraSendiri->lastItem() ?? 0 }}</strong>
                                    of
                                    <strong>{{ $lokerMitraSendiri->total() }}</strong>
                                    entries
                                </div>

                                <nav>
                                    <ul class="pagination mb-0">
                                        {{-- Previous Page Link --}}
                                        @if ($lokerMitraSendiri->onFirstPage())
                                            <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                                        @else
                                            <li class="page-item"><a class="page-link"
                                                    href="{{ $lokerMitraSendiri->previousPageUrl() }}"
                                                    rel="prev">&laquo;</a></li>
                                        @endif

                                        {{-- Pagination Elements --}}
                                        @foreach ($lokerMitraSendiri->getUrlRange(1, $lokerMitraSendiri->lastPage()) as $page => $url)
                                            @if ($page == $lokerMitraSendiri->currentPage())
                                                <li class="page-item active"><span
                                                        class="page-link">{{ $page }}</span>
                                                </li>
                                            @elseif ($page == 1 || $page == $lokerMitraSendiri->lastPage() || abs($page - $lokerMitraSendiri->currentPage()) <= 1)
                                                <li class="page-item"><a class="page-link"
                                                        href="{{ $url }}">{{ $page }}</a></li>
                                            @elseif ($page == 2 || $page == $lokerMitraSendiri->lastPage() - 1)
                                                <li class="page-item disabled"><span class="page-link">...</span></li>
                                            @endif
                                        @endforeach

                                        {{-- Next Page Link --}}
                                        @if ($lokerMitraSendiri->hasMorePages())
                                            <li class="page-item"><a class="page-link"
                                                    href="{{ $lokerMitraSendiri->nextPageUrl() }}"
                                                    rel="next">&raquo;</a>
                                            </li>
                                        @else
                                            <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                                        @endif
                                    </ul>
                                </nav>
                            </div>

                            {{-- Loker Mitra Lain --}}
                            <hr class="container my-5">
                            <h5 class="mb-4 fw-bold">Lowongan Kerja dari Mitra Lain</h5>
                            <div class="sertif-container mt-4">
                                <div class="row row-cols-1 row-cols-md-3 g-4">
                                    @forelse($lokerMitraLain as $loker)
                                        <div class="col">
                                            <div class="card loker-card h-100 border-0 shadow-sm">
                                                <img src="{{ asset('storage/' . $loker->gambar) }}"
                                                    class="card-img-top object-fit-cover"
                                                    alt="Foto Loker {{ $loker->nama_perusahaan }}"
                                                    style="height: 200px; object-fit: cover;">

                                                <div class="card-body d-flex flex-column">
                                                    <h6 class="fw-semibold mb-1">{{ $loker->posisi }}</h6>

                                                    <ul class="list-unstyled text-muted sertif-info mb-3 small">
                                                        <li><i class="bi bi-building"></i>
                                                            {{ $loker->nama_perusahaan }}
                                                        </li>
                                                        <li><i class="bi bi-briefcase"></i> {{ $loker->tipe }}</li>
                                                        <li><i class="bi bi-cash-stack"></i> {{ $loker->gaji }}</li>
                                                    </ul>
                                                </div>
                                            </div>
                                        </div>
                                    @empty
                                        <p class="text-muted">Belum ada Lowongan Kerja dari mitra lain.</p>
                                    @endforelse
                                </div>
                            </div>

                        </div>
                    </div>

                    {{-- Modal Tambah --}}
                    <div class="modal fade" id="modalTambah" tabindex="-1" aria-labelledby="modalTambahLokerLabel"
                        aria-hidden="true">
                        <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                            <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                                <div class="row g-0">

                                    <!-- Ilustrasi Kiri -->
                                    <div
                                        class="col-md-5 d-none d-md-flex bg-light flex-column justify-content-center align-items-center p-4">
                                        <h5 class="text-primary fw-bold">Tambah Lowongan Kerja</h5>
                                        <p class="text-muted text-center px-3">
                                            Tambahkan informasi lowongan pekerjaan baru untuk alumni sesuai kebutuhan
                                            mitra
                                            perusahaan.
                                        </p>
                                    </div>

                                    <!-- Form Tambah -->
                                    <div class="col-md-7 bg-white">
                                        <form action="{{ route('mitra.loker.store') }}" method="POST"
                                            enctype="multipart/form-data" class="p-4" id="formTambahLoker">
                                            @csrf
                                            <input type="hidden" name="action" value="store">

                                            <div class="modal-header border-0 pb-0">
                                                <h5 class="modal-title fw-bold" id="modalTambahLokerLabel">
                                                    <i class="bi bi-briefcase-fill me-2 text-primary"></i>Tambah Loker
                                                </h5>
                                            </div>

                                            <div class="modal-body overflow-auto" style="max-height: 65vh;">
                                                <div class="mb-3">
                                                    <label class="form-label">Nama Perusahaan</label>
                                                    <div class="input-group">
                                                        <span class="input-group-text"><i
                                                                class="bi bi-building"></i></span>
                                                        <input type="text" name="nama_perusahaan"
                                                            class="form-control" required>
                                                    </div>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Posisi</label>
                                                    <div class="input-group">
                                                        <span class="input-group-text"><i
                                                                class="bi bi-person-badge"></i></span>
                                                        <input type="text" name="posisi" class="form-control"
                                                            required>
                                                    </div>
                                                </div>

                                                <div class="row">
                                                    <div class="col-md-6 mb-3">
                                                        <label class="form-label">Lokasi</label>
                                                        <input type="text" name="lokasi" class="form-control"
                                                            required>
                                                    </div>
                                                    <div class="col-md-6 mb-3">
                                                        <label class="form-label">Tipe Kerja</label>
                                                        <select name="tipe" class="form-select" required>
                                                            <option value="">Pilih Tipe</option>
                                                            <option value="Full Time">Full Time</option>
                                                            <option value="Part Time">Part Time</option>
                                                            <option value="Remote">Remote</option>
                                                        </select>
                                                    </div>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Pendidikan Minimal</label>
                                                    <select name="pendidikan" class="form-select" required>
                                                        <option disabled>Pilih Pendidikan</option>
                                                        <option value="SMA/SMK">SMA/SMK</option>
                                                        <option value="D3">D3</option>
                                                        <option value="S1">S1</option>
                                                    </select>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Rentang Gaji (Rp)</label>
                                                    <div class="row">
                                                        <div class="col">
                                                            <input type="text" class="form-control"
                                                                id="gaji_min_view" placeholder="Minimum" required
                                                                oninput="syncGaji(this, 'min')">
                                                            <input type="hidden" name="gaji_min" id="gaji_min">
                                                        </div>
                                                        <div class="col">
                                                            <input type="text" class="form-control"
                                                                id="gaji_max_view" placeholder="Maksimum" required
                                                                oninput="syncGaji(this, 'max')">
                                                            <input type="hidden" name="gaji_max" id="gaji_max">
                                                        </div>
                                                    </div>
                                                </div>


                                                <div class="mb-3">
                                                    <label class="form-label">Deskripsi</label>
                                                    <textarea name="deskripsi" class="form-control" rows="3" placeholder="Deskripsi singkat tentang pekerjaan..."
                                                        required></textarea>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Skill yang di butuhkan</label>
                                                    <input name="skills" class="form-control" placeholder="Skill yang dibutukan"
                                                        required></input>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Upload Gambar</label>
                                                    <input type="file" name="gambar" class="form-control"
                                                        accept=".jpg,.jpeg,.png" required>
                                                </div>
                                            </div>

                                            <div class="modal-footer border-0 pt-0">
                                                <button type="button" class="btn btn-outline-secondary"
                                                    data-bs-dismiss="modal">Batal</button>
                                                <button type="submit" class="btn btn-primary">Simpan
                                                    Lowongan</button>
                                            </div>
                                        </form>
                                    </div>

                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Modal Edit (Versi Lebih Rapi dan Ilustratif) -->
                    <div class="modal fade" id="editLokerModal" tabindex="-1" aria-labelledby="editLokerModalLabel"
                        aria-hidden="true">
                        <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                            <div class="modal-content edit-modal-content border-0 shadow-lg">
                                <form id="editForm" method="POST" enctype="multipart/form-data"
                                    onsubmit="return validateGajiEdit()">
                                    @csrf
                                    @method('PUT')
                                    <input type="hidden" name="id" id="edit_id">

                                    <div class="modal-header bg-primary text-white">
                                        <h5 class="modal-title fw-semibold">
                                            <i class="bi bi-pencil-square me-2"></i>Edit Lowongan Kerja
                                        </h5>
                                        <button type="button" class="btn-close btn-close-white"
                                            data-bs-dismiss="modal" aria-label="Close"></button>
                                    </div>

                                    <div class="modal-body px-4 pt-4" style="max-height: 75vh; overflow-y: auto;">
                                        <div class="row gx-5 gy-4">
                                            <!-- Kiri -->
                                            <div class="col-lg-6">
                                                <div class="mb-3">
                                                    <label class="form-label">Nama Perusahaan</label>
                                                    <input type="text" class="form-control"
                                                        id="edit_nama_perusahaan" name="nama_perusahaan" required>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Deskripsi</label>
                                                    <textarea class="form-control" id="edit_deskripsi" name="deskripsi" rows="5" required></textarea>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Skill yang dibutuhkan</label>
                                                    <input class="form-control" id="edit_skills" name="skills" required></input>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Pendidikan Minimal</label>
                                                    <select name="pendidikan" id="edit_pendidikan"
                                                        class="form-select" required>
                                                        <option disabled>Pilih Pendidikan</option>
                                                        <option value="SMA/SMK">SMA/SMK</option>
                                                        <option value="D3">D3</option>
                                                        <option value="S1">S1</option>
                                                    </select>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Upload Gambar (Opsional)</label>
                                                    <input type="file" class="form-control" id="edit_gambar"
                                                        name="gambar" accept=".jpg,.jpeg,.png"
                                                        onchange="previewImageEdit(this)">
                                                    <img id="imagePreviewEdit"
                                                        class="img-fluid mt-2 rounded d-none border"
                                                        style="max-height: 150px;" alt="Preview Gambar">
                                                </div>
                                            </div>

                                            <!-- Kanan -->
                                            <div class="col-lg-6">
                                                <div class="mb-3">
                                                    <label class="form-label">Posisi</label>
                                                    <input type="text" class="form-control" id="edit_posisi"
                                                        name="posisi" required>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Lokasi</label>
                                                    <input type="text" class="form-control" id="edit_lokasi"
                                                        name="lokasi" required>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Tipe Pekerjaan</label>
                                                    <select class="form-select" name="tipe" id="edit_tipe"
                                                        required>
                                                        <option value="Part Time">Part Time</option>
                                                        <option value="Full Time">Full Time</option>
                                                        <option value="Remote">Remote</option>
                                                    </select>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Rentang Gaji (Rp)</label>
                                                    <div class="row gx-2">
                                                        <div class="col">
                                                            <input type="text" class="form-control"
                                                                id="edit_gaji_min_view" placeholder="Minimum" required
                                                                oninput="formatRupiah(this)">
                                                            <input type="hidden" name="gaji_min" id="edit_gaji_min">
                                                        </div>
                                                        <div class="col">
                                                            <input type="text" class="form-control"
                                                                id="edit_gaji_max_view" placeholder="Maksimum"
                                                                required oninput="formatRupiah(this)">
                                                            <input type="hidden" name="gaji_max" id="edit_gaji_max">
                                                        </div>
                                                    </div>
                                                    <small id="gajiErrorEdit" class="text-danger d-none">Gaji maksimum
                                                        harus lebih besar dari minimum.</small>
                                                </div>

                                                <div class="mb-3">
                                                    <label class="form-label">Status Loker</label>
                                                    <select class="form-select" name="status" id="edit_status"
                                                        required>
                                                        <option value="buka">Buka</option>
                                                        <option value="tutup">Tutup</option>
                                                    </select>
                                                </div>
                                            </div>
                                        </div>
                                    </div>

                                    <div class="modal-footer bg-light border-top-0 px-4 py-3">
                                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">
                                            <i class="bi bi-x-circle me-1"></i> Batal
                                        </button>
                                        <button type="button" class="btn btn-primary" onclick="confirmEdit()">
                                            <i class="bi bi-save me-1"></i> Perbarui
                                        </button>
                                    </div>
                                </form>
                            </div>
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
            function syncGaji(input, type) {
                // Hilangkan karakter non-digit
                let raw = input.value.replace(/\D/g, '');
                let formatted = new Intl.NumberFormat('id-ID').format(raw);
                input.value = formatted;

                // Set ke input hidden
                if (type === 'min') {
                    document.getElementById('gaji_min').value = raw;
                } else if (type === 'max') {
                    document.getElementById('gaji_max').value = raw;
                }
            }
        </script>
        <script>
            document.getElementById('formTambahLoker').addEventListener('submit', function(e) {
                const min = parseInt(document.getElementById('gaji_min').value || '0');
                const max = parseInt(document.getElementById('gaji_max').value || '0');

                if (min > max) {
                    e.preventDefault();
                    Swal.fire({
                        icon: 'error',
                        title: 'Rentang Gaji Tidak Valid',
                        text: 'Gaji minimum tidak boleh lebih besar dari gaji maksimum.',
                    });
                }
            });
        </script>
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const downloadButtons = document.querySelectorAll('.btnDownloadPeserta');

                downloadButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        const lokerId = this.getAttribute('data-id');
                        window.location.href =
                            `/mitra/loker/${lokerId}/peserta/export`;
                    });
                });
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
            function confirmDelete(id) {
                Swal.fire({
                    title: 'Hapus Loker?',
                    text: "Data loker yang dihapus tidak dapat dikembalikan.",
                    icon: 'warning',
                    showCancelButton: true,
                    confirmButtonText: 'Ya, Hapus!',
                    cancelButtonText: 'Batal',
                    confirmButtonColor: '#d33',
                    cancelButtonColor: '#6c757d'
                }).then((result) => {
                    if (result.isConfirmed) {
                        document.getElementById('deleteForm' + id).submit();
                    }
                });
            }

            function confirmTambah() {
                Swal.fire({
                    title: 'Simpan Loker?',
                    text: "Pastikan semua informasi sudah lengkap dan benar.",
                    icon: 'question',
                    showCancelButton: true,
                    confirmButtonColor: '#198754',
                    cancelButtonColor: '#6c757d',
                    confirmButtonText: 'Ya, Simpan',
                    cancelButtonText: 'Batal'
                }).then((result) => {
                    if (result.isConfirmed) {
                        document.getElementById('tambahForm').submit();
                    }
                });
            }

            function formatRupiah(input) {
                let value = input.value.replace(/\D/g, '');
                input.value = new Intl.NumberFormat('id-ID').format(value);
            }

            function formatRupiah(input) {
                let value = input.value.replace(/\D/g, ''); // Hapus semua karakter non-digit
                let formatted = value.replace(/\B(?=(\d{3})+(?!\d))/g, '.'); // Tambah titik setiap 3 digit dari belakang
                input.value = formatted;
            }
        </script>
        <script>
            function formatRupiah(input) {
                let raw = input.value.replace(/\D/g, '');
                input.value = new Intl.NumberFormat('id-ID').format(raw);

                if (input.id === 'edit_gaji_min_view') {
                    document.getElementById('edit_gaji_min').value = raw;
                } else if (input.id === 'edit_gaji_max_view') {
                    document.getElementById('edit_gaji_max').value = raw;
                }
            }

            function validateGajiEdit() {
                const min = parseInt(document.getElementById('edit_gaji_min').value || '0');
                const max = parseInt(document.getElementById('edit_gaji_max').value || '0');
                const error = document.getElementById('gajiErrorEdit');

                if (min > max) {
                    error.classList.remove('d-none');
                    return false;
                } else {
                    error.classList.add('d-none');
                    return true;
                }
            }

            function confirmEdit() {
                if (!validateGajiEdit()) return;

                Swal.fire({
                    title: 'Perbarui Lowongan?',
                    text: 'Apakah kamu yakin ingin memperbarui data ini?',
                    icon: 'question',
                    showCancelButton: true,
                    confirmButtonText: 'Ya, Perbarui',
                    cancelButtonText: 'Batal',
                    confirmButtonColor: '#0d6efd',
                    cancelButtonColor: '#6c757d',
                }).then((result) => {
                    if (result.isConfirmed) {
                        document.getElementById('editForm').submit();
                    }
                });
            }

            function previewImageEdit(input) {
                const file = input.files[0];
                const preview = document.getElementById('imagePreviewEdit');

                if (file) {
                    const reader = new FileReader();
                    reader.onload = function(e) {
                        preview.src = e.target.result;
                        preview.classList.remove('d-none');
                    };
                    reader.readAsDataURL(file);
                }
            }

            function openEditLokerModal(data) {
                document.getElementById('edit_id').value = data.id;
                document.getElementById('editForm').action = '/mitra/update-loker/' + data.id;
                document.getElementById('edit_nama_perusahaan').value = data.nama_perusahaan;
                document.getElementById('edit_posisi').value = data.posisi;
                document.getElementById('edit_lokasi').value = data.lokasi;
                document.getElementById('edit_tipe').value = data.tipe;
                document.getElementById('edit_pendidikan').value = data.pendidikan;
                document.getElementById('edit_deskripsi').value = data.deskripsi;
                document.getElementById('edit_skills').value = data.skills;
                document.getElementById('edit_status').value = data.status;

                // Gaji
                if (data.gaji) {
                    let match = data.gaji.match(/Rp\s?([\d.]+)\s?-\s?Rp\s?([\d.]+)/);
                    if (match) {
                        let gajiMin = match[1].replace(/\./g, '');
                        let gajiMax = match[2].replace(/\./g, '');
                        document.getElementById('edit_gaji_min').value = gajiMin;
                        document.getElementById('edit_gaji_max').value = gajiMax;
                        document.getElementById('edit_gaji_min_view').value = new Intl.NumberFormat('id-ID').format(gajiMin);
                        document.getElementById('edit_gaji_max_view').value = new Intl.NumberFormat('id-ID').format(gajiMax);
                    }
                }

                // Preview gambar
                if (data.foto) {
                    const preview = document.getElementById('imagePreviewEdit');
                    preview.src = data.foto;
                    preview.classList.remove('d-none');
                }
            }
        </script>


</body>

</html>
