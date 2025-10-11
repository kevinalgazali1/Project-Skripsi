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
            overflow-x: hidden;
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

        .topbar {
            background-color: #fff;
            border-bottom: 1px solid #dee2e6;
            padding: 15px 30px;
        }

        .nav-title {
            font-weight: 600;
            font-size: 18px;
        }

        .users-table {
            border-radius: 0.5rem;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            background-color: #ffffff;
            width: 100%;
        }

        .users-table thead {
            background-color: #f1f3f5;
        }

        .users-table thead th {
            font-weight: 600;
            color: #343a40;
            text-transform: uppercase;
            font-size: 0.8rem;
        }

        .users-table tbody tr {
            transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.3s ease;
            background-color: #fff !important;
        }

        .users-table tbody tr:hover {
            transform: scale(1.02);
            background-color: #ffffff !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            color: #fff;
            z-index: 1;
            position: relative;
        }


        .users-table td,
        .users-table th {
            vertical-align: middle;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
        }

        .users-table .btn-group .btn {
            padding: 0.3rem 0.5rem;
            font-size: 0.75rem;
            border-radius: 0.375rem;
        }

        .users-table .badge {
            font-size: 0.7rem;
            padding: 0.35em 0.5em;
            border-radius: 0.375rem;
        }

        .dropdown-menu .dropdown-item:hover {
            background-color: #f8f9fa;
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

        .users-table tbody tr:hover {
            background-color: #f8f9fa;
            transition: background-color 0.2s;
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
            <div class="topbar d-flex justify-content-between bg-primary align-items-center text-light">
                <span class="nav-title">{{ $title ?? 'Dashboard' }}</span>
                <span class="text-kight">Hi, Admin</span>
            </div>

            <div class="container-fluid content py-4 px-0">

                <div class="row g-4 mb-2 px-4">
                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-primary bg-opacity-10 text-primary rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-people fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Total Alumni</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalAlumni }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-success bg-opacity-10 text-success rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-building fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Total Sekolah</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalSekolah }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-info bg-opacity-10 text-info rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-list-ul fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Total Jurusan</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalJurusan }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <form method="GET" action="" class="mb-4 px-4">
                    <div class="row g-3 mb-3">
                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Cari Nama Alumni</label>
                            <input type="text" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="search" value="{{ request('search') }}" placeholder="Nama Alumni">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Asal Sekolah</label>
                            <input type="text" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="asal_sekolah" value="{{ request('asal_sekolah') }}"
                                placeholder="Asal Sekolah">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Jurusan</label>
                            <input type="text" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="jurusan" value="{{ request('jurusan') }}" placeholder="Jurusan Sekolah">
                        </div>
                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Status</label>
                            <select class="form-select rounded-3 shadow-sm border border-light-subtle" name="status">
                                <option value="">Semua</option>
                                <option value="Bekerja" {{ request('status') == 'Bekerja' ? 'selected' : '' }}>Bekerja
                                </option>
                                <option value="Wirausaha" {{ request('status') == 'Wirausaha' ? 'selected' : '' }}>
                                    Wirausaha</option>
                                <option value="Kuliah" {{ request('status') == 'Kuliah' ? 'selected' : '' }}>Kuliah
                                </option>
                                <option value="Tidak Bekerja"
                                    {{ request('status') == 'Tidak Bekerja' ? 'selected' : '' }}>Tidak Bekerja</option>
                            </select>
                        </div>
                    </div>

                    <div class="row g-3 mb-3">
                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Tahun Kelulusan</label>
                            <input type="number" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="tahun_kelulusan" value="{{ request('tahun_kelulusan') }}"
                                placeholder="Tahun Kelulusan">
                        </div>

                        <div class="col-md-3">
                            <label class="form-label fw-medium text-secondary small mb-1">Nama Universitas</label>
                            <input type="text" class="form-control rounded-3 shadow-sm border border-light-subtle"
                                name="nama_universitas" value="{{ request('nama_universitas') }}"
                                placeholder="Nama Universitas">
                        </div>

                        <div class="col-md-6">
                            <label class="form-label fw-medium text-secondary small mb-1">Rentang Waktu</label>
                            <div class="input-group shadow-sm rounded-3 border border-light-subtle overflow-hidden">
                                <input type="date" class="form-control border-0" name="tanggal_mulai"
                                    value="{{ request('tanggal_mulai') }}">
                                <span
                                    class="input-group-text bg-body-tertiary border-0 text-secondary small">s/d</span>
                                <input type="date" class="form-control border-0" name="tanggal_selesai"
                                    value="{{ request('tanggal_selesai') }}">
                            </div>
                        </div>
                    </div>

                    <div class="row g-3 align-items-center mb-3">
                        <div class="col-md-12 d-flex justify-content-end gap-2">
                            <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-search me-1"></i> Terapkan
                            </button>
                            <a href="{{ route('admin.usersalumni') }}"
                                class="btn btn-outline-secondary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-arrow-clockwise me-1"></i> Reset
                            </a>
                            <a href="{{ route('export.alumni.csv') }}"
                                class="btn btn-success rounded-pill px-4 shadow-sm">
                                <i class="bi bi-download me-1"></i> CSV
                            </a>
                        </div>
                    </div>
                </form>

                <div class="table-responsive users-table-wrapper">
                    <div id="tableLoadingUsers" class="d-none text-center py-5">
                        <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>

                    <table class="table users-table table-hover align-middle mb-0" id="usersTable">
                        <thead class="table-light text-center text-uppercase small">
                            <tr>
                                <th>No</th>
                                <th>Nama Lengkap</th>
                                <th>NIK</th>
                                <th>Tahun Kelulusan</th>
                                <th>Asal Sekolah</th>
                                <th>Jurusan Sekolah</th>
                                <th>Jenis Kelamin</th>
                                <th>Status</th>
                                <th>Universitas</th>
                                <th>Aksi</th>
                            </tr>
                        </thead>
                        <tbody>
                            @forelse ($profiles as $index => $p)
                                <tr class="align-middle">
                                    <td class="text-center fw-semibold text-secondary">{{ $index + 1 }}</td>

                                    <td class="fw-semibold text-dark">{{ $p->nama_lengkap }}</td>

                                    <td class="text-center text-muted small">{{ $p->nik }}</td>

                                    <td class="text-center">{{ $p->tahun_kelulusan ?? '-' }}</td>

                                    <td class="fw-normal text-dark">{{ $p->asal_sekolah }}</td>

                                    <td class="fw-normal text-dark">{{ $p->jurusan_sekolah }}</td>

                                    <td class="text-center">{{ $p->jenis_kelamin }}</td>

                                    <td class="text-center">
                                        @php
                                            $status = strtolower($p->status_saat_ini);
                                            $badgeClass = match ($status) {
                                                'bekerja' => 'bg-success-subtle text-success',
                                                'wirausaha' => 'bg-warning-subtle text-warning',
                                                'kuliah' => 'bg-primary-subtle text-primary',
                                                'tidak bekerja' => 'bg-danger-subtle text-danger',
                                                default => 'bg-secondary-subtle text-secondary',
                                            };
                                        @endphp

                                        <span class="badge rounded-pill px-3 py-2 fw-medium {{ $badgeClass }}">
                                            {{ $p->status_saat_ini }}
                                        </span>

                                    </td>

                                    <td class="fw-normal text-dark">{{ $p->nama_universitas }}</td>

                                    <td class="text-center">
                                        <div class="dropdown">
                                            <button class="btn btn-sm btn-primary border shadow-sm dropdown-toggle"
                                                type="button" data-bs-toggle="dropdown" aria-expanded="false">
                                                Aksi
                                            </button>
                                            <ul class="dropdown-menu dropdown-menu-end shadow-sm">
                                                <li>
                                                    <button class="dropdown-item d-flex align-items-center gap-2"
                                                        data-bs-toggle="modal"
                                                        data-bs-target="#lihat{{ $p->id }}">
                                                        <i class="bi bi-eye text-primary"></i> Lihat
                                                    </button>
                                                </li>
                                            </ul>
                                        </div>
                                    </td>
                                </tr>

                            @empty
                                <tr>
                                    <td colspan="10" class="text-center py-4 text-muted">
                                        <i class="bi bi-exclamation-circle me-2"></i> Data tidak ditemukan
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>

                    <div class="d-flex justify-content-between align-items-center mt-3 flex-wrap gap-2 px-4 mb-5">
                        <div class="small text-muted">
                            Showing
                            <strong>{{ $profiles->firstItem() ?? 0 }}</strong>
                            to
                            <strong>{{ $profiles->lastItem() ?? 0 }}</strong>
                            of
                            <strong>{{ $profiles->total() }}</strong>
                            entries
                        </div>

                        <nav>
                            <ul class="pagination mb-0">
                                {{-- Previous Page Link --}}
                                @if ($profiles->onFirstPage())
                                    <li class="page-item disabled"><span class="page-link">&laquo;</span></li>
                                @else
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $profiles->previousPageUrl() }}" rel="prev">&laquo;</a>
                                    </li>
                                @endif

                                {{-- Pagination Elements --}}
                                @foreach ($profiles->getUrlRange(1, $profiles->lastPage()) as $page => $url)
                                    @if ($page == $profiles->currentPage())
                                        <li class="page-item active"><span
                                                class="page-link">{{ $page }}</span>
                                        </li>
                                    @elseif ($page == 1 || $page == $profiles->lastPage() || abs($page - $profiles->currentPage()) <= 1)
                                        <li class="page-item"><a class="page-link"
                                                href="{{ $url }}">{{ $page }}</a></li>
                                    @elseif ($page == 2 || $page == $profiles->lastPage() - 1)
                                        <li class="page-item disabled"><span class="page-link">...</span></li>
                                    @endif
                                @endforeach

                                {{-- Next Page Link --}}
                                @if ($profiles->hasMorePages())
                                    <li class="page-item"><a class="page-link"
                                            href="{{ $profiles->nextPageUrl() }}" rel="next">&raquo;</a></li>
                                @else
                                    <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                                @endif
                            </ul>
                        </nav>
                    </div>
                </div>

            </div>

            <!-- Modal Lihat Data -->
            @foreach ($profiles as $index => $p)
                <div class="modal fade" id="lihat{{ $p->id }}" tabindex="-1"
                    aria-labelledby="lihatModalLabel{{ $p->id }}" aria-hidden="true">
                    <div class="modal-dialog modal-lg modal-dialog-centered modal-dialog-scrollable">
                        <div class="modal-content rounded-4 shadow">

                            <div class="modal-header bg-primary text-white rounded-top">
                                <h5 class="modal-title" id="lihatModalLabel{{ $p->id }}">Detail Alumni -
                                    {{ $p->nama_lengkap }}</h5>
                            </div>

                            <div class="modal-body">

                                <div class="row g-4">

                                    <!-- Foto Profil -->
                                    <div class="col-md-4 text-center">
                                        <img src="{{ asset('storage/' . $p->foto_profil) }}"
                                            alt="Foto {{ $p->nama_lengkap }}"
                                            class="img-fluid rounded shadow-sm mb-2"
                                            style="max-height: 250px; object-fit: cover;">
                                        <div class="fw-medium mt-2 text-dark small">{{ $p->nama_lengkap }}</div>
                                        <div class="fw-medium mt-2 text-dark small">{{ $p->user->email ?? '-' }}</div>
                                    </div>

                                    <!-- Data Detail -->
                                    <div class="col-md-8">
                                        <table class="table table-borderless small">
                                            <tr>
                                                <th>Nama Lengkap</th>
                                                <td>: {{ $p->nama_lengkap }}</td>
                                            </tr>
                                            <tr>
                                                <th>NIK</th>
                                                <td>: {{ $p->nik }}</td>
                                            </tr>
                                            <tr>
                                                <th>Tahun Kelulusan</th>
                                                <td>: {{ $p->tahun_kelulusan }}</td>
                                            </tr>
                                            <tr>
                                                <th>Asal Sekolah</th>
                                                <td>: {{ $p->asal_sekolah }} </td>
                                            </tr>
                                            <tr>
                                                <th>Jurusan Sekolah</th>
                                                <td>: {{ $p->jurusan_sekolah }} </td>
                                            </tr>
                                            <tr>
                                                <th>Jenis Kelamin</th>
                                                <td>: {{ $p->jenis_kelamin }} </td>
                                            </tr>
                                            <tr>
                                                <th>Tanggal Lahir</th>
                                                <td>: {{ \Carbon\Carbon::parse($p->tanggal_lahir)->format('d F Y') }}
                                                </td>
                                            </tr>
                                            <tr>
                                                <th>Provinsi</th>
                                                <td>: {{ $p->provinsi }} </td>
                                            </tr>
                                            <tr>
                                                <th>Kota</th>
                                                <td>: {{ $p->kota }} </td>
                                            </tr>
                                            <tr>
                                                <th>Alamat</th>
                                                <td>: {{ $p->alamat }} </td>
                                            </tr>
                                            <tr>
                                                <th>Status Saat Ini</th>
                                                <td>: {{ $p->status_saat_ini }}</td>
                                            </tr>
                                            @if ($p->bidang_pekerjaan)
                                                <tr>
                                                    <th>Bidang Pekerjaan</th>
                                                    <td>: {{ $p->bidang_pekerjaan }}</td>
                                                </tr>
                                            @endif
                                            @if ($p->sertifikasi_terakhir)
                                                <tr>
                                                    <th>Sertifikasi Terakhir</th>
                                                    <td>: {{ $p->sertifikasi_terakhir }}</td>
                                                </tr>
                                            @endif
                                            @if ($p->kesesuaian_sertifikasi)
                                                <tr>
                                                    <th>Kesesuaian Sertifikasi</th>
                                                    <td>: {{ $p->kesesuaian_sertifikasi }}</td>
                                                </tr>
                                            @endif
                                            @if ($p->nama_universitas)
                                                <tr>
                                                    <th>Universitas</th>
                                                    <td>: {{ $p->nama_universitas }} </td>
                                                </tr>
                                                <tr>
                                                    <th>Jurusan Kuliah</th>
                                                    <td>: {{ $p->jurusan_universitas }}</td>
                                                </tr>
                                            @endif
                                        </table>
                                    </div>

                                </div>

                            </div>

                            <div class="modal-footer bg-light rounded-bottom">
                                <button type="button" class="btn btn-secondary"
                                    data-bs-dismiss="modal">Tutup</button>
                            </div>

                        </div>
                    </div>
                </div>
            @endforeach
        </div>
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
