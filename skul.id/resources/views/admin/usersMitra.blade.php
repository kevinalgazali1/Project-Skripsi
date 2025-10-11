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

        .topbar {
            background-color: #fff;
            border-bottom: 1px solid #dee2e6;
            padding: 15px 30px;
        }

        .nav-title {
            font-weight: 600;
            font-size: 18px;
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

            <div class="container-fluid py-4 px-0">

                {{-- Statistik --}}
                <div class="row g-4 mb-2 px-4">
                    <div class="col-md-4">
                        <div class="card shadow-sm border-0 rounded-3 h-100">
                            <div class="card-body d-flex align-items-center gap-3">
                                <div class="d-flex justify-content-center align-items-center bg-primary bg-opacity-10 text-primary rounded-circle"
                                    style="width: 60px; height: 60px;">
                                    <i class="bi bi-people fs-4"></i>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1">Total Mitra</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalMitra }}</h3>
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
                                    <h6 class="text-muted mb-1">Total Bidang</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalBidang }}</h3>
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
                                    <h6 class="text-muted mb-1">Total Kota</h6>
                                    <h3 class="mb-0 fw-bold">{{ $totalKota }}</h3>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {{-- Filter Form --}}
                <form method="GET" class="row g-3 mb-4 px-4">
                    <div class="col-md-3">
                        <input type="text" name="search" class="form-control" placeholder="Cari nama perusahaan..."
                            value="{{ request('search') }}">
                    </div>
                    <div class="col-md-3">
                        <select class="form-select rounded-3" name="kategori">
                            <option disabled {{ request('kategori') ? '' : 'selected' }}>Pilih Kategori Mitra</option>
                            <option value="kampus" {{ request('kategori') == 'kampus' ? 'selected' : '' }}>Kampus
                            </option>
                            <option value="sekolah" {{ request('kategori') == 'sekolah' ? 'selected' : '' }}>Sekolah
                            </option>
                            <option value="instansi Pemerintah"
                                {{ request('kategori') == 'instansi Pemerintah' ? 'selected' : '' }}>Instansi
                                Pemerintah</option>
                            <option value="swasta" {{ request('kategori') == 'swasta' ? 'selected' : '' }}>Swasta
                            </option>
                        </select>
                    </div>

                    <div class="col-md-3">
                        <input type="text" name="bidang" class="form-control" placeholder="Bidang usaha..."
                            value="{{ request('bidang') }}">
                    </div>
                    <div class="col-md-3">
                        <input type="text" name="kota" class="form-control" placeholder="Kota..."
                            value="{{ request('kota') }}">
                    </div>
                    <div class="row g-3 align-items-center mb-3">
                        <div class="col-md-12 d-flex justify-content-end gap-2">
                            <button type="submit" class="btn btn-primary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-search me-1"></i> Terapkan
                            </button>
                            <a href="{{ route('admin.usersmitra') }}"
                                class="btn btn-outline-secondary rounded-pill px-4 shadow-sm">
                                <i class="bi bi-arrow-clockwise me-1"></i> Reset
                            </a>

                            <button type="button" class="btn btn-success rounded-pill px-4 shadow-sm"
                                data-bs-toggle="modal" data-bs-target="#modalTambah">+ Tambah
                                Mitra</button>
                        </div>
                    </div>
                </form>

                <div class="table-responsive users-table-wrapper px-2">
                    <div id="tableLoadingUsers" class="d-none text-center py-5">
                        <div class="spinner-border text-primary" role="status" style="width: 3rem; height: 3rem;">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                    </div>

                    <table class="table users-table table-hover align-middle mb-0" id="usersMitraTable">
                        <thead class="table-light text-center text-uppercase small">
                            <tr>
                                <th>No</th>
                                <th>Nama Instansi</th>
                                <th>Penanggung Jawab</th>
                                <th>Bidang Industri</th>
                                <th>Kategori</th>
                                <th>Provinsi</th>
                                <th>Kota</th>
                                <th>Alamat</th>
                                <th>Aksi</th>
                            </tr>
                        </thead>
                        <tbody>
                            @forelse ($profiles as $index => $m)
                                <tr class="align-middle text-center">
                                    <td class="fw-semibold text-secondary">{{ $profiles->firstItem() + $index }}</td>

                                    <td class="fw-semibold text-dark text-start">{{ $m->nama_instansi }}</td>

                                    <td class="fw-normal text-dark">{{ $m->penanggung_jawab }}</td>

                                    <td>
                                        <span class="badge rounded-pill bg-info-subtle text-info px-3 py-2 fw-medium">
                                            {{ $m->bidang_industri }}
                                        </span>
                                    </td>

                                    <td>
                                        @php
                                            $kategori = strtolower($m->kategori);
                                            $badgeClass = match ($kategori) {
                                                'perusahaan' => 'bg-primary-subtle text-primary',
                                                'industri' => 'bg-success-subtle text-success',
                                                'organisasi' => 'bg-warning-subtle text-warning',
                                                default => 'bg-secondary-subtle text-secondary',
                                            };
                                        @endphp
                                        <span class="badge rounded-pill px-3 py-2 fw-medium {{ $badgeClass }}">
                                            {{ ucfirst($m->kategori) }}
                                        </span>
                                    </td>

                                    <td class="text-capitalize">{{ $m->provinsi }}</td>
                                    <td class="text-capitalize">{{ $m->kota }}</td>
                                    <td class="text-start">{{ $m->alamat }}</td>

                                    <td>
                                        <form action="{{ route('admin.usersmitra') }}" method="POST"
                                            onsubmit="return confirm('Yakin ingin menghapus mitra ini?')"
                                            class="d-inline">
                                            @csrf
                                            <input type="hidden" name="action" value="delete">
                                            <input type="hidden" name="id" value="{{ $m->id }}">
                                            <button type="submit"
                                                class="btn btn-sm btn-outline-danger d-flex align-items-center gap-1">
                                                <i class="bi bi-trash-fill"></i> Hapus
                                            </button>
                                        </form>
                                    </td>
                                </tr>
                            @empty
                                <tr>
                                    <td colspan="9" class="text-center py-4 text-muted">
                                        <i class="bi bi-exclamation-circle me-2"></i> Data mitra tidak ditemukan.
                                    </td>
                                </tr>
                            @endforelse
                        </tbody>
                    </table>
                </div>
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
                                <li class="page-item"><a class="page-link" href="{{ $profiles->previousPageUrl() }}"
                                        rel="prev">&laquo;</a>
                                </li>
                            @endif

                            {{-- Pagination Elements --}}
                            @foreach ($profiles->getUrlRange(1, $profiles->lastPage()) as $page => $url)
                                @if ($page == $profiles->currentPage())
                                    <li class="page-item active"><span class="page-link">{{ $page }}</span>
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
                                <li class="page-item"><a class="page-link" href="{{ $profiles->nextPageUrl() }}"
                                        rel="next">&raquo;</a></li>
                            @else
                                <li class="page-item disabled"><span class="page-link">&raquo;</span></li>
                            @endif
                        </ul>
                    </nav>
                </div>
            </div>
        </div>

        <!-- Modal Tambah -->
        <div class="modal fade" id="modalTambah" tabindex="-1">
            <div class="modal-dialog modal-lg modal-dialog-centered">
                <div class="modal-content">
                    <form action="{{ route('admin.usersmitra.store') }}" method="POST">
                        @csrf
                        <input type="hidden" name="role" value="mitra">
                        <div class="modal-header">
                            <h5 class="modal-title">Pendaftaran Mitra</h5>
                            <button class="btn-close" data-bs-dismiss="modal"></button>
                        </div>
                        <div class="modal-body">
                            <div class="mb-3 position-relative">
                                <label class="form-label">Nomor HP Telkomsel</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-phone"></i></span>
                                    <input type="text" class="form-control" placeholder="08xxxxxxxxxx"
                                        name="no_hp" required>
                                </div>
                                @if ($errors->has('no_hp'))
                                    <div class="alert alert-danger mt-2">
                                        {{ $errors->first('no_hp') }}
                                    </div>
                                @endif
                            </div>

                            <div class="mb-3 position-relative">
                                <label class="form-label">Password</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-lock"></i></span>
                                    <input type="password" class="form-control" placeholder="Password"
                                        name="password" id="password" required>
                                    <span class="input-group-text" onclick="togglePassword('password', 'icon1')"
                                        style="cursor: pointer;">
                                        <i class="bi bi-eye-slash" id="icon1"></i>
                                    </span>
                                </div>
                            </div>

                            <div class="mb-3 position-relative">
                                <label class="form-label">Konfirmasi Password</label>
                                <div class="input-group">
                                    <span class="input-group-text"><i class="bi bi-lock-fill"></i></span>
                                    <input type="password" class="form-control" placeholder="Konfirmasi Password"
                                        name="password_confirmation" id="passwordConfirm" required>
                                    <span class="input-group-text"
                                        onclick="togglePassword('passwordConfirm', 'icon2')" style="cursor: pointer;">
                                        <i class="bi bi-eye-slash" id="icon2"></i>
                                    </span>
                                </div>
                            </div>
                        </div>
                        <div class="modal-footer">
                            <button class="btn btn-primary">Daftar</button>
                            <button class="btn btn-secondary" data-bs-dismiss="modal">Tutup</button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    <script>
        function togglePassword(fieldId, iconId) {
            const field = document.getElementById(fieldId);
            const icon = document.getElementById(iconId);
            if (field.type === "password") {
                field.type = "text";
                icon.classList.remove("bi-eye-slash");
                icon.classList.add("bi-eye");
            } else {
                field.type = "password";
                icon.classList.remove("bi-eye");
                icon.classList.add("bi-eye-slash");
            }
        }
    </script>
    <script>
        @if (session('success'))
            Swal.fire({
                icon: 'success',
                title: 'Berhasil!',
                text: '{{ session('success') }}',
                confirmButtonColor: '#3085d6'
            });
        @elseif (session('error'))
            Swal.fire({
                icon: 'error',
                title: 'Gagal!',
                text: '{{ session('error') }}',
                confirmButtonColor: '#d33'
            });
        @endif
    </script>
    @if (session('success'))
        <script>
            Swal.fire({
                icon: 'success',
                title: 'Berhasil!',
                text: '{{ session('success') }}',
                confirmButtonColor: '#3085d6'
            }).then(() => {
                const modal = bootstrap.Modal.getInstance(document.getElementById('modalTambah'));
                if (modal) modal.hide();
            });
        </script>
    @endif
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
