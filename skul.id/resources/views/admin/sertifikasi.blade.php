<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ $title ?? 'Super Admin' }}</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap" rel="stylesheet" />
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">

    <style>
        body {
            background-color: #f4f6f9;
            font-family: 'Poppins', sans-serif;
        }

        .sidebar {
            min-width: 250px;
            min-height: 100vh;
            background-color: #fff;
            border-right: 1px solid #e1e1e1;
        }

        .sidebar .nav-link {
            color: #333;
            font-weight: 500;
            padding: 12px 20px;
            border-radius: 8px;
            margin-bottom: 8px;
        }

        .sidebar .nav-link.active,
        .sidebar .nav-link:hover {
            background-color: #0d6efd;
            color: #fff !important;
        }

        .card-stat {
            border: none;
            border-radius: 16px;
            background: linear-gradient(135deg, var(--color), #96c93d);
            color: #fff;
            transition: transform 0.2s ease;
        }

        .card-stat:hover {
            transform: translateY(-3px);
        }

        .mitra-card img {
            width: 100px;
            height: 100px;
            object-fit: cover;
        }

        .topbar {
            background-color: #fff;
            border-bottom: 1px solid #dee2e6;
            padding: 15px 30px;
        }

        .nav-title {
            font-weight: 600;
            font-size: 18px;
        }

        h2,
        h4,
        h6 {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }

        .card h6 {
            font-weight: 600;
        }

        .soft-table {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.03);
            background-color: #f7f9fc;
        }

        .soft-table thead {
            background-color: #f7f9fc;
            border-bottom: 1px solid #e0e6ed;
        }

        .soft-table th,
        .soft-table td {
            vertical-align: middle;
            padding: 0.75rem;
        }

        .soft-table tbody tr:hover {
            background-color: #f5faff;
            transition: background-color 0.3s ease;
        }

        .badge-soft {
            display: inline-block;
            padding: 4px 10px;
            font-size: 0.75rem;
            font-weight: 500;
            border-radius: 6px;
        }

        .badge-aktif {
            background-color: #e6f4ea;
            color: #2e7d32;
        }

        .badge-nonaktif {
            background-color: #fcecec;
            color: #c62828;
        }

        .text-muted-small {
            font-size: 0.85rem;
            color: #6c757d;
        }

        .btn-icon {
            border: none;
            background: transparent;
            color: #6c757d;
            font-size: 1rem;
        }

        .btn-icon:hover {
            color: #0d6efd;
        }

        .rounded-soft {
            border-radius: 10px !important;
        }

        .modal-content {
            border-radius: 1rem;
            overflow: hidden;
        }

        .detail th {
            position: sticky;
            top: 0;
            background-color: #f8f9fa;
        }

        .img-fluid {
            max-height: 250px;
            object-fit: cover;
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

        .dropdown-menu .dropdown-item:hover {
            background-color: #f8f9fa;
        }

        .sertifikasi-table-wrapper.fade {
            opacity: 0.5;
            transition: opacity 0.3s ease;
        }

        .sertifikasi-table {
            border-radius: 0.5rem;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
            width: 100%;
        }

        .sertifikasi-table thead {
            background-color: #f1f3f5;
        }

        .sertifikasi-table thead th {
            font-weight: 600;
            color: #343a40;
            text-transform: uppercase;
            font-size: 0.85rem;
        }

        .sertifikasi-table tbody tr {
            text-align: center;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
            background-color: #fff !important;
        }

        .sertifikasi-table tbody tr:hover {
            transform: scale(1.02);
            background-color: #ffffff !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            color: #fff;
            z-index: 1;
            position: relative;
        }


        .sertifikasi-table td,
        .sertifikasi-table th {
            vertical-align: middle;
            padding: 0.75rem 1rem;
            font-size: 0.8rem;
        }

        .sertifikasi-table .btn-group .btn {
            padding: 0.3rem 0.5rem;
            font-size: 0.75rem;
            border-radius: 0.375rem;
        }

        .sertifikasi-table .badge {
            font-size: 0.7rem;
            padding: 0.35em 0.5em;
            border-radius: 0.375rem;
        }
    </style>
</head>

<body>

    <div class="d-flex">
        <!-- Sidebar -->
        <div class="sidebar p-4 shadow-sm">
            <h4 class="text-primary mb-4">Super Admin</h4>
            <div class="nav flex-column">
                <a href="{{ route('admin.dashboard') }}"
                    class="nav-link {{ request()->routeIs('admin.dashboard') ? 'active' : '' }}">
                    <i class="bi bi-speedometer2 me-2"></i> Dashboard
                </a>
                <a href="{{ route('admin.sertifikasi') }}"
                    class="nav-link {{ request()->routeIs('admin.sertifikasi') ? 'active' : '' }}">
                    <i class="bi bi-award-fill me-2"></i> Sertifikasi
                </a>
                <a href="{{ route('admin.pelatihan') }}"
                    class="nav-link {{ request()->routeIs('admin.pelatihan') ? 'active' : '' }}">
                    <i class="bi bi-mortarboard-fill me-2"></i> Pelatihan
                </a>
                <a href="{{ route('admin.loker') }}"
                    class="nav-link {{ request()->routeIs('admin.loker') ? 'active' : '' }}">
                    <i class="bi bi-briefcase-fill me-2"></i> Loker
                </a>
                <a href="{{ route('admin.artikel') }}" class="nav-link"
                    class="nav-link {{ request()->routeIs('admin.artikel') ? 'active' : '' }}">
                    <i class="bi bi-file-earmark-text-fill me-2"></i> Artikel
                </a>
                <div class="nav-item">
                    <a class="nav-link d-flex justify-content-between align-items-center {{ request()->is('admin/users*') ? 'active' : '' }}"
                        data-bs-toggle="collapse" href="#submenuUsers" role="button"
                        aria-expanded="{{ request()->is('admin/users*') ? 'true' : 'false' }}"
                        aria-controls="submenuUsers">
                        <span><i class="bi bi-people-fill me-2"></i> Users</span>
                        <i class="bi bi-chevron-down small"></i>
                    </a>
                    <div class="collapse {{ request()->is('admin/users*') ? 'show' : '' }}" id="submenuUsers">
                        <a href="{{ route('admin.usersalumni') }}"
                            class="nav-link ps-4 {{ request()->routeIs('admin.usersalumni') ? 'active' : '' }}">
                            <i class="bi bi-person-lines-fill me-2"></i> Alumni
                        </a>
                        <a href="{{ route('admin.usersmitra') }}"
                            class="nav-link ps-4 {{ request()->routeIs('admin.usersmitra') ? 'active' : '' }}">
                            <i class="bi bi-building me-2"></i> Mitra
                        </a>
                    </div>
                </div>
                <form id="logout-form" action="{{ route('logout') }}" method="POST" class="mt-auto">
                    @csrf
                    <a href="#" class="nav-link d-flex align-items-center text-danger fw-semibold mb-4"
                        onclick="confirmLogout(event)">
                        <i class="bi bi-box-arrow-right me-2"></i> Logout
                    </a>
                </form>
            </div>
            <div class="mt-auto pt-5 text-muted small">
                &copy; {{ date('Y') }} Super Admin
            </div>
        </div>

        <!-- Main Content -->
        <div class="flex-grow-1">
            <!-- Topbar -->
            <div class="topbar d-flex justify-content-between align-items-center bg-primary text-light p-3">
                <span class="nav-title">{{ $title ?? 'Dashboard' }}</span>
                <span class="text-light">Hi, Admin</span>
            </div>

            <div class="container-fluid content py-4 px-0">
                <div class="row g-4 px-4">
                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-primary bg-opacity-10 text-primary rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-bar-chart-line fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Total Sertifikasi</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalSertifikasi ?? 0 }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-success bg-opacity-10 text-success rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-people-fill fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Pengguna Terdaftar</h6>
                                    <h3 class="mb-0 fw-bold">{{ $persentaseUserTerdaftar ?? 0 }}%</h3>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-info bg-opacity-10 text-info rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-check-circle fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Sertifikasi Selesai</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalSertifikasiSelesai ?? 0 }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <form method="GET" action="" class="mb-4 px-4">
                <div class="row g-3 mb-3">
                    <div class="col-md-3">
                        <label class="form-label fw-semibold small text-muted">Nama Mitra</label>
                        <input type="text" class="form-control" name="mitra" placeholder="Nama Mitra"
                            value="{{ request('mitra') }}">
                    </div>
                    <div class="col-md-3">
                        <label class="form-label fw-semibold small text-muted">Judul Sertifikasi</label>
                        <input type="text" class="form-control" name="judul" placeholder="Judul Sertifikasi"
                            value="{{ request('judul') }}">
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-semibold small text-muted">Kota</label>
                        <input type="text" class="form-control" name="kota" placeholder="Kota"
                            value="{{ request('kota') }}">
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-semibold small text-muted">Bulan Mulai</label>
                        <select name="bulan_mulai" class="form-select">
                            <option value="" disabled {{ request('bulan_mulai') ? '' : 'selected' }}>Pilih Bulan
                            </option>
                            @foreach ([
        '01' => 'Januari',
        '02' => 'Februari',
        '03' => 'Maret',
        '04' => 'April',
        '05' => 'Mei',
        '06' => 'Juni',
        '07' => 'Juli',
        '08' => 'Agustus',
        '09' => 'September',
        '10' => 'Oktober',
        '11' => 'November',
        '12' => 'Desember',
    ] as $val => $label)
                                <option value="{{ $val }}"
                                    {{ request('bulan_mulai') == $val ? 'selected' : '' }}>
                                    {{ $label }}
                                </option>
                            @endforeach
                        </select>
                    </div>

                    <div class="col-md-2">
                        <label class="form-label fw-semibold small text-muted">Tahun Mulai</label>
                        <input type="number" class="form-control" name="tahun_mulai" placeholder="Tahun Mulai"
                            value="{{ request('tahun_mulai') }}">
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-semibold small text-muted">Status Selesai</label>
                        <select class="form-select" name="status_selesai">
                            <option value="">Semua</option>
                            <option value="selesai" {{ request('status_selesai') == 'selesai' ? 'selected' : '' }}>
                                Selesai
                            </option>
                            <option value="belum" {{ request('status_selesai') == 'belum' ? 'selected' : '' }}>Belum
                                Selesai</option>
                        </select>
                    </div>
                    <div class="col-md-2">
                        <label class="form-label fw-semibold small text-muted">Status</label>
                        <select class="form-select" name="status">
                            <option value="">Semua Status</option>
                            <option value="Gratis" {{ request('status') == 'Gratis' ? 'selected' : '' }}>Gratis
                            </option>
                            <option value="Berbayar" {{ request('status') == 'Berbayar' ? 'selected' : '' }}>Berbayar
                            </option>
                        </select>
                    </div>
                    <div class="col d-flex justify-content-end gap-2 align-items-end">
                        <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                            <i class="bi bi-search me-1"></i> Terapkan
                        </button>
                        <a href="{{ route('admin.sertifikasi') }}"
                            class="btn btn-outline-secondary rounded-pill px-4 shadow-sm">
                            <i class="bi bi-arrow-clockwise me-1"></i> Reset
                        </a>
                        <a href="{{ route('admin.sertifikasi.export', request()->query()) }}"
                            class="btn btn-outline-success rounded-pill px-4 shadow-sm">
                            <i class="bi bi-file-earmark-excel me-1"></i> Excel
                        </a>
                        <button type="button" class="btn btn-success rounded-pill px-4 shadow-sm"
                            data-bs-toggle="modal" data-bs-target="#modalTambah">
                            <i class="bi bi-plus-circle me-1"></i> Tambah
                        </button>
                    </div>
                </div>
            </form>



            <div class="container py-4">
                <div class="table-responsive sertifikasi-table-wrapper">
                    <table class="table sertifikasi-table table-hover align-middle mb-0" id="sertifikasiTable">
                        <thead class="table-light text-center text-uppercase small">
                            <tr>
                                <th>No</th>
                                <th>Mitra</th>
                                <th>Judul Sertifikasi</th>
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
                            @forelse ($sertifikasis as $s)
                                <tr>
                                    <td class="text-center">{{ $loop->iteration }}</td>
                                    <td class="text-center">
                                        <div class="fw-semibold">{{ $s->user->mitraProfile->nama_instansi ?? '-' }}
                                        </div>
                                    </td>
                                    <td>
                                        <div class="fw-semibold">{{ $s->judul_sertifikasi }}</div>
                                    </td>
                                    <td class="text-center">{{ $s->kota }}</td>
                                    <td class="text-center">{{ $s->daftarSertifikasis->count() }}</td>
                                    <td class="text-center">{{ $s->jumlah_asal_sekolah }}</td>
                                    <td class="text-center">{{ $s->jumlah_jurusan }}</td>
                                    <td class="text-center">{{ $s->jumlah_bekerja_dan_usaha }}</td>
                                    <td class="text-center">{{ $s->jumlah_tidak_bekerja }}</td>
                                    <td class="text-center">{{ $s->jumlah_kuliah }}</td>
                                    <td class="text-center">
                                        <div class="dropdown">
                                            <button class="btn btn-sm btn-primary dropdown-toggle" type="button"
                                                data-bs-toggle="dropdown" aria-expanded="false">
                                                Aksi
                                            </button>
                                            <ul class="dropdown-menu dropdown-menu-end">
                                                <li>
                                                    <button class="dropdown-item" data-bs-toggle="modal"
                                                        data-bs-target="#modalLihat{{ $s->id }}">
                                                        <i class="bi bi-eye me-2"></i> Lihat
                                                    </button>
                                                </li>
                                                @php
                                                    $sudahMulai = \Carbon\Carbon::parse($s->tanggal_mulai)->isPast();
                                                @endphp
                                                @if (!$sudahMulai)
                                                    <li>
                                                        <button class="dropdown-item" data-bs-toggle="modal"
                                                            data-bs-target="#modalEdit{{ $s->id }}">
                                                            <i class="bi bi-pencil me-2"></i> Edit
                                                        </button>
                                                    </li>
                                                @endif
                                                <li>
                                                    <form action="{{ route('admin.sertifikasi') }}" method="POST"
                                                        class="d-inline">
                                                        @csrf
                                                        <input type="hidden" name="action" value="delete">
                                                        <input type="hidden" name="id"
                                                            value="{{ $s->id }}">
                                                        <button type="button" class="dropdown-item text-danger"
                                                            onclick="confirmDelete('{{ $s->id }}')">
                                                            <i class="bi bi-trash me-2"></i> Hapus
                                                        </button>
                                                    </form>
                                                </li>
                                            </ul>
                                        </div>
                                    </td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="11" class="text-center text-muted py-4">Belum ada data sertifikasi.
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>

                    <div class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                        <div class="small text-muted">
                            Showing
                            <strong>{{ $sertifikasis->firstItem() ?? 0 }}</strong>
                            to
                            <strong>{{ $sertifikasis->lastItem() ?? 0 }}</strong>
                            of
                            <strong>{{ $sertifikasis->total() }}</strong>
                            entries
                        </div>

                        <nav>
                            <ul class="pagination mb-0">
                                {{-- Previous Page Link --}}
                                @if ($sertifikasis->onFirstPage())
                                    <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                                @else
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $sertifikasis->previousPageUrl() }}" rel="prev">&laquo;</a>
                                    </li>
                                @endif

                                {{-- Pagination Elements --}}
                                @foreach ($sertifikasis->getUrlRange(1, $sertifikasis->lastPage()) as $page => $url)
                                    @if ($page == $sertifikasis->currentPage())
                                        <li class="page-item active"><span
                                                class="page-link">{{ $page }}</span>
                                        </li>
                                    @elseif ($page == 1 || $page == $sertifikasis->lastPage() || abs($page - $sertifikasis->currentPage()) <= 1)
                                        <li class="page-item"><a class="page-link"
                                                href="{{ $url }}">{{ $page }}</a></li>
                                    @elseif ($page == 2 || $page == $sertifikasis->lastPage() - 1)
                                        <li class="page-item disabled"><span class="page-link">...</span></li>
                                    @endif
                                @endforeach

                                {{-- Next Page Link --}}
                                @if ($sertifikasis->hasMorePages())
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $sertifikasis->nextPageUrl() }}" rel="next">&raquo;</a></li>
                                @else
                                    <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                                @endif
                            </ul>
                        </nav>
                    </div>
                </div>
            </div>

            {{-- Modal Tambah --}}
            <div class="modal fade" id="modalTambah" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                    <div class="modal-content border-0 shadow-lg rounded-4 overflow-hidden">
                        <div class="row g-0">
                            <!-- Ilustrasi -->
                            <div class="col-md-5 d-none d-md-block bg-light position-relative">
                                <div class="h-100 d-flex flex-column justify-content-center align-items-center p-4">
                                    <h5 class="text-primary fw-bold">Tambah Sertifikasi Baru</h5>
                                    <p class="text-muted text-center px-3">
                                        Publikasikan program sertifikasi Anda agar alumni dapat
                                        mengembangkan keterampilan dan
                                        meningkatkan daya saing mereka.
                                    </p>

                                </div>
                            </div>
                            <!-- Form -->
                            <div class="col-md-7 bg-white" style="max-height: 90vh; overflow-y: auto;">
                                <form action="{{ route('admin.sertifikasi.store') }}" method="POST"
                                    enctype="multipart/form-data" class="p-4" id="tambahForm">
                                    @csrf
                                    <div class="modal-header border-0 pb-0">
                                        <h5 class="modal-title fw-bold" id="sertifikasiModalLabel">
                                            <i class="bi bi-award-fill me-2 text-primary"></i>Tambah
                                            Sertifikasi
                                        </h5>
                                    </div>
                                    <div class="modal-body">
                                        <div class="mb-3">
                                            <label for="judul_sertifikasi" class="form-label">Judul
                                                Sertifikasi</label>
                                            <div class="input-group">
                                                <span class="input-group-text"><i class="bi bi-card-text"></i></span>
                                                <input type="text" class="form-control" id="judul_sertifikasi"
                                                    name="judul_sertifikasi"
                                                    placeholder="Contoh: Sertifikasi Digital Marketing" required>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <label for="deskripsi" class="form-label">Deskripsi</label>
                                            <textarea class="form-control" id="deskripsi" name="deskripsi" rows="3"
                                                placeholder="Jelaskan sertifikasi ini secara singkat..." required></textarea>
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
                                                <input type="date" class="form-control" id="tanggal_selesai"
                                                    name="tanggal_selesai" required>
                                            </div>
                                        </div>
                                        <div class="row">
                                            <div class="col-md-6 mb-3">
                                                <label for="kota" class="form-label">Kota
                                                    Penyelenggaraan</label>
                                                <input type="text" class="form-control" id="kota"
                                                    name="kota" placeholder="Contoh: Jakarta" required>
                                            </div>
                                            <div class="col-md-6 mb-3">
                                                <label for="tempat" class="form-label">Tempat
                                                    Penyelenggaraan</label>
                                                <input type="text" class="form-control" id="tempat"
                                                    name="tempat" placeholder="Contoh: Gedung Balai Kartini"
                                                    required>
                                            </div>
                                        </div>
                                        <div class="mb-3">
                                            <label for="status" class="form-label">Status
                                                Sertifikasi</label>
                                            <select class="form-select" id="status" name="status" required
                                                onchange="toggleBiaya()">
                                                <option value="Berbayar">Berbayar</option>
                                                <option value="Gratis" selected>Gratis</option>
                                            </select>
                                        </div>
                                        <div class="mb-3" id="biayaGroup">
                                            <label for="biaya" class="form-label">Biaya Sertifikasi
                                                (Rp)</label>
                                            <input type="string" class="form-control" id="biaya" name="biaya"
                                                placeholder="Contoh: 250000" oninput="formatRupiah(this)" required>
                                        </div>
                                        <div class="mb-3" id="rekeningGroup" style="display: none;">
                                            <label for="nomor_rekening" class="form-label">Nomor Rekening</label>
                                            <input type="text" class="form-control" id="nomor_rekening"
                                                name="nomor_rekening"
                                                placeholder="Contoh: 1234567890 a.n PT Sertifikasi BCA">
                                        </div>
                                        <div class="mb-3">
                                            <label for="foto_sertifikasi" class="form-label">Upload
                                                Foto Sertifikasi</label>
                                            <input type="file" class="form-control" id="foto_sertifikasi"
                                                name="foto_sertifikasi" accept=".jpg,.jpeg,.png" required>
                                        </div>
                                        <div class="modal-footer border-0 pt-0">
                                            <button type="button" class="btn btn-outline-secondary"
                                                data-bs-dismiss="modal">Batal</button>
                                            <button type="button" class="btn btn-primary"
                                                onclick="confirmTambah()">Simpan
                                                Sertifikasi</button>
                                        </div>
                                    </div>
                                </form>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            @foreach ($sertifikasis as $s)
                <div class="modal fade" id="modalLihat{{ $s->id }}" tabindex="-1" aria-hidden="true">
                    <div class="modal-dialog modal-dialog-centered modal-xl">
                        <div class="modal-content shadow-lg rounded-4">
                            <div class="modal-header bg-primary text-white">
                                <h5 class="modal-title">Detail Sertifikasi</h5>
                            </div>
                            <div class="modal-body px-4 py-4">
                                <div class="row g-4">
                                    {{-- Kolom Kiri: Detail Sertifikasi --}}
                                    <div class="col-lg-6">
                                        @if ($s->foto_sertifikasi)
                                            <div class="mb-3">
                                                <img src="{{ asset('storage/' . $s->foto_sertifikasi) }}"
                                                    alt="Foto Sertifikasi"
                                                    class="img-fluid rounded-3 shadow-sm border w-100"
                                                    style="max-height: 280px; object-fit: cover;">
                                            </div>
                                        @endif

                                        <h4 class="fw-bold mb-2">{{ $s->judul_sertifikasi }}</h4>

                                        {{-- Scrollable Deskripsi --}}
                                        <div class="mb-3">
                                            <div style="max-height: 180px; overflow-y: auto;">
                                                <p class="mb-0" style="white-space: pre-line;">
                                                    {!! nl2br(e($s->deskripsi)) !!}</p>
                                            </div>
                                        </div>

                                    </div>


                                    {{-- Kolom Kanan: Daftar Peserta --}}
                                    <div class="col-lg-6">

                                        <div class="mb-2"><strong>📍 Lokasi:</strong> {{ $s->tempat }},
                                            {{ $s->kota }}
                                        </div>
                                        <div class="mb-2"><strong>📅 Tanggal:</strong> {{ $s->tanggal_mulai }} s/d
                                            {{ $s->tanggal_selesai }}</div>
                                        <div class="mb-2">
                                            <strong>📌 Status:</strong>
                                            <span
                                                class="badge bg-{{ $s->status == 'Aktif' ? 'success' : 'secondary' }}">
                                                {{ $s->status }}
                                            </span>
                                        </div>
                                        <div class="mb-2">
                                            <strong>💳 Harga Sertifikasi:</strong>
                                            @php
                                                $rawBiaya = str_replace('.', '', $s->biaya); // Hapus titik dari string
                                                $cleanBiaya = is_numeric($rawBiaya) ? (int) $rawBiaya : 0;
                                            @endphp

                                            {{ $cleanBiaya > 0 ? 'Rp ' . number_format($cleanBiaya, 0, ',', '.') : 'Gratis' }}

                                        </div>
                                        <div class="d-flex justify-content-between align-items-center mb-3 mt-5">
                                            <h5 class="fw-semibold mb-0">👥 Peserta Terdaftar</h5>

                                            {{-- Tombol Download --}}
                                            <button data-id="{{ $s->id }}"
                                                class="btn btn-sm btn-outline-success d-flex align-items-center btnDownloadPeserta">
                                                <i class="bi bi-download me-1"></i> Download Peserta
                                            </button>

                                        </div>

                                        @if ($s->daftarSertifikasis && $s->daftarSertifikasis->count())
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
                                                        @foreach ($s->daftarSertifikasis as $peserta)
                                                            <tr class="text-center">
                                                                <td>{{ $loop->iteration }}</td>
                                                                <td class="text-truncate" style="max-width: 120px;">
                                                                    {{ $peserta->nama_lengkap }}</td>
                                                                <td class="text-truncate" style="max-width: 150px;">
                                                                    {{ $peserta->email }}</td>
                                                                <td>{{ $peserta->no_hp }}</td>
                                                                <td class="text-center">
                                                                    @if ($peserta->bukti_transfer)
                                                                        <img src="{{ asset($peserta->bukti_transfer) }}"
                                                                            alt="Bukti Transfer" class="img-thumbnail"
                                                                            style="max-height: 50px; cursor: pointer;"
                                                                            data-bs-toggle="modal"
                                                                            data-bs-target="#imageModal"
                                                                            data-image="{{ asset($peserta->bukti_transfer) }}"
                                                                            data-close-parent="#modalLihat{{ $s->id }}"
                                                                            onclick="this.classList.add('was-open')">
                                                                    @else
                                                                        <span class="text-muted">Tidak ada</span>
                                                                    @endif
                                                                </td>

                                                            </tr>
                                                        @endforeach
                                                    </tbody>
                                                </table>
                                            </div>
                                        @else
                                            <p class="text-muted mt-2">Belum ada peserta yang mendaftar.</p>
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

            <div class="modal fade" id="imageModal" tabindex="-1" aria-hidden="true">
                <div class="modal-dialog modal-dialog-centered modal-lg">
                    <div class="modal-content border-0">
                        <div class="modal-header">
                            <h5 class="modal-title">Pratinjau Bukti Transfer</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal"
                                aria-label="Close"></button>
                        </div>
                        <div class="modal-body text-center">
                            <img id="modalImagePreview" src="" class="img-fluid rounded shadow"
                                alt="Preview">
                        </div>
                    </div>
                </div>
            </div>

            {{-- Loop untuk Modal Edit --}}
            @foreach ($sertifikasis as $s)
                <div class="modal fade" id="modalEdit{{ $s->id }}" tabindex="-1"
                    aria-labelledby="modalEditLabel{{ $s->id }}" aria-hidden="true">
                    <div class="modal-dialog modal-xl modal-dialog-centered modal-dialog-scrollable">
                        <div class="modal-content edit-modal-content">
                            <form id="editForm" method="POST"
                                action="{{ url('/admin/update-sertifikasi/' . $s->id) }}"
                                enctype="multipart/form-data">
                                @csrf
                                @method('PUT')
                                <input type="hidden" name="action" value="update">
                                <input type="hidden" name="id" value="{{ $s->id }}">

                                <div class="modal-header border-0">
                                    <h5 class="modal-title fw-semibold text-primary-emphasis">
                                        <i class="bi bi-pencil-square me-2"></i>Edit Sertifikasi
                                    </h5>
                                </div>

                                <div class="modal-body">
                                    <div class="row gx-4 gy-3">
                                        <!-- Left Section -->
                                        <div class="col-lg-6">
                                            <div class="form-group mb-3">
                                                <label class="form-label">Judul Sertifikasi</label>
                                                <input type="text" class="form-control edit-input"
                                                    name="judul_sertifikasi" value="{{ $s->judul_sertifikasi }}"
                                                    required>
                                            </div>

                                            <div class="form-group mb-3">
                                                <label class="form-label">Deskripsi</label>
                                                <textarea class="form-control edit-input" name="deskripsi" rows="4" required>{{ $s->deskripsi }}</textarea>
                                            </div>

                                            <div class="row">
                                                <div class="col-md-6 mb-3">
                                                    <label class="form-label">Tanggal Mulai</label>
                                                    <input type="date" class="form-control edit-input"
                                                        name="tanggal_mulai" value="{{ $s->tanggal_mulai }}"
                                                        required>
                                                </div>
                                                <div class="col-md-6 mb-3">
                                                    <label class="form-label">Tanggal Selesai</label>
                                                    <input type="date" class="form-control edit-input"
                                                        name="tanggal_selesai" value="{{ $s->tanggal_selesai }}"
                                                        required>
                                                </div>
                                            </div>
                                        </div>

                                        <!-- Right Section -->
                                        <div class="col-lg-6">
                                            <div class="form-group mb-3">
                                                <label class="form-label">Kota</label>
                                                <input type="text" class="form-control edit-input" name="kota"
                                                    value="{{ $s->kota }}" required>
                                            </div>

                                            <div class="form-group mb-3">
                                                <label class="form-label">Tempat</label>
                                                <input type="text" class="form-control edit-input" name="tempat"
                                                    value="{{ $s->tempat }}" required>
                                            </div>

                                            <div class="form-group mb-3">
                                                <label class="form-label">Status Sertifikasi</label>
                                                <select class="form-select edit-input" name="status"
                                                    onchange="toggleBiayaField({{ $s->id }})" required>
                                                    <option value="Berbayar"
                                                        {{ $s->status == 'Berbayar' ? 'selected' : '' }}>Berbayar
                                                    </option>
                                                    <option value="Gratis"
                                                        {{ $s->status == 'Gratis' ? 'selected' : '' }}>Gratis</option>
                                                </select>
                                            </div>

                                            <div class="form-group mb-3" id="edit_biaya_wrapper{{ $s->id }}">
                                                <label class="form-label">Biaya (Rp)</label>
                                                <input type="string" class="form-control edit-input" name="biaya"
                                                    oninput="formatRupiah(this)" value="{{ $s->biaya }}">
                                            </div>

                                            <div class="form-group mb-3"
                                                id="edit_rekening_wrapper{{ $s->id }}">
                                                <label class="form-label">Nomor Rekening</label>
                                                <input type="text" class="form-control edit-input"
                                                    name="nomor_rekening" value="{{ $s->nomor_rekening }}">
                                            </div>

                                            <div class="form-group">
                                                <label class="form-label">Upload Foto Sertifikasi (Opsional)</label>
                                                <input type="file" class="form-control edit-input"
                                                    name="foto_sertifikasi" accept=".jpg,.jpeg,.png"
                                                    onchange="previewImage(this, 'preview_foto_{{ $s->id }}')">
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                <div class="modal-footer border-top-0">
                                    <button type="button" class="btn btn-light"
                                        data-bs-dismiss="modal">Batal</button>
                                    <button type="button" class="btn btn-primary"
                                        onclick="confirmEdit()">Perbarui</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </div>
            @endforeach


        </div>

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
        function confirmDelete(id) {
            Swal.fire({
                title: 'Hapus Sertifikasi?',
                text: "Data sertifikasi yang dihapus tidak dapat dikembalikan.",
                icon: 'warning',
                showCancelButton: true,
                confirmButtonText: 'Ya, Hapus!',
                cancelButtonText: 'Batal',
                confirmButtonColor: '#d33',
                cancelButtonColor: '#6c757d'
            }).then((result) => {
                if (result.isConfirmed) {
                    // Buat form dinamis & submit
                    const form = document.createElement('form');
                    form.method = 'POST';
                    form.action = "{{ route('admin.sertifikasi') }}";

                    form.innerHTML = `
                          @csrf
                          <input type="hidden" name="action" value="delete">
                          <input type="hidden" name="id" value="${id}">
                        `;
                    document.body.appendChild(form);
                    form.submit();
                }
            });
        }

        function confirmTambah() {
            Swal.fire({
                title: 'Simpan Sertifikasi?',
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

        function confirmEdit() {
            Swal.fire({
                title: 'Perbarui Sertifikasi?',
                text: "Pastikan semua data sudah benar.",
                icon: 'question',
                showCancelButton: true,
                confirmButtonColor: '#0d6efd',
                cancelButtonColor: '#6c757d',
                confirmButtonText: 'Ya, perbarui!',
                cancelButtonText: 'Batal'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('editForm').submit();
                }
            });
        }
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

            downloadButtons.forEach(function(btn) {
                btn.addEventListener('click', function() {
                    const sertifikasiId = this.getAttribute('data-id');
                    if (sertifikasiId) {
                        window.location.href =
                            `/public/admin/sertifikasi/${sertifikasiId}/peserta/export`;
                    }
                });
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
            @foreach ($sertifikasis as $s)
                toggleBiayaField({{ $s->id }});
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
