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

        <!-- Logout Confirmation Modal -->
        <div class="modal fade" id="logoutModal" tabindex="-1" aria-labelledby="logoutModalLabel" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content rounded-3">
                    <div class="modal-header">
                        <h5 class="modal-title" id="logoutModalLabel">Konfirmasi Logout</h5>
                        <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Tutup"></button>
                    </div>
                    <div class="modal-body">
                        Apakah Anda yakin ingin logout?
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Batal</button>
                        <form id="logout-form" action="{{ route('logout') }}" method="POST" class="m-0">
                            @csrf
                            <button type="submit" class="btn btn-danger">Logout</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Content -->
        <div class="flex-grow-1">
            <!-- Topbar -->
            <div class="topbar d-flex justify-content-between align-items-center bg-primary text-light p-3">
                <span class="nav-title">{{ $title ?? 'Dashboard' }}</span>
                <span class="text-light">Hi, Admin</span>
            </div>

            <!-- Page Content -->
            <main class="container-fluid py-5 px-4">
                <button class="btn btn-primary mb-5" data-bs-toggle="modal" data-bs-target="#editAccountModal">Edit
                    Akun</button>

                <!-- Stat Cards -->
                <div class="row g-4 mb-5">
                    @foreach ([['Total Sertifikasi', $totalSertifikasi, 'bi-award', 'bg-primary'], ['Total Pelatihan', $totalPelatihan, 'bi-mortarboard', 'bg-success'], ['Total Loker', $totalLoker, 'bi-briefcase', 'bg-warning']] as [$label, $value, $icon, $bg])
                        <div class="col-md-4">
                            <div class="d-flex align-items-center p-4 bg-white rounded shadow-sm border h-100">
                                <div class="me-3">
                                    <span
                                        class="d-inline-flex align-items-center justify-content-center rounded-circle {{ $bg }} bg-opacity-25 text-dark"
                                        style="width: 48px; height: 48px;">
                                        <i class="bi {{ $icon }} fs-4"></i>
                                    </span>
                                </div>
                                <div>
                                    <h6 class="text-muted mb-1 small">{{ $label }}</h6>
                                    <h4 class="mb-0 fw-bold">{{ $value }}</h4>
                                </div>
                            </div>
                        </div>
                    @endforeach
                </div>

                <!-- Extra Stats -->
                <div class="row g-4 mb-5">
                    @foreach ([['Jumlah Peserta', $jumlahPeserta, 'bi-people', 'text-primary'], ['Jumlah Sekolah', $jumlahSekolah, 'bi-building', 'text-success'], ['Jumlah Mitra', $jumlahMitra, 'bi-buildings', 'text-warning']] as [$title, $value, $icon, $color])
                        <div class="col-md-4">
                            <div class="p-4 bg-white border rounded shadow-sm text-center h-100">
                                <i class="bi {{ $icon }} fs-2 mb-2 {{ $color }}"></i>
                                <h6 class="text-muted">{{ $title }}</h6>
                                <h4 class="fw-bold">{{ $value }}</h4>
                            </div>
                        </div>
                    @endforeach
                </div>

                <!-- Daftar Mitra -->
                <div class="mb-4 d-flex justify-content-between align-items-center">
                    <h4 class="fw-semibold fs-4">Daftar Mitra</h4>
                    <a href="{{ route('admin.usersmitra') }}" class="btn btn-sm btn-outline-primary">Lihat Semua</a>
                </div>
                <div class="row g-4">
                    @forelse ($mitras as $mitra)
                        <div class="col-md-4">
                            <div class="card border-0 shadow-sm h-100">
                                <div class="card-body d-flex flex-column align-items-center text-center">
                                    @if ($mitra->logo)
                                        <img src="{{ asset('storage/' . $mitra->logo) }}"
                                            alt="Logo {{ $mitra->nama_instansi }}" class="rounded-circle mb-3"
                                            style="width: 72px; height: 72px; object-fit: cover;">
                                    @else
                                        <div class="mb-3">
                                            <i class="bi bi-building fs-1 text-muted"></i>
                                        </div>
                                    @endif
                                    <h4 class="fw-semibold mb-1">{{ $mitra->nama_instansi }}</h4>
                                    <h6 class="fw-semibold mb-1">{{ $mitra->penanggung_jawab }}</h6>
                                </div>
                            </div>
                        </div>
                    @empty
                        <div class="col-12">
                            <div class="alert alert-secondary text-center">Belum ada mitra yang terdaftar.</div>
                        </div>
                    @endforelse
                </div>
            </main>
            <div class="modal fade" id="editAccountModal" tabindex="-1" aria-labelledby="editAccountModalLabel"
                aria-hidden="true">
                <div class="modal-dialog">
                    <form method="POST" id="editForm" action="{{ route('admin.account.update') }}">
                        @csrf
                        @method('PUT')

                        <div class="modal-content">
                            <div class="modal-header">
                                <h5 class="modal-title" id="editAccountModalLabel">Edit Akun</h5>
                                <button type="button" class="btn-close" data-bs-dismiss="modal"
                                    aria-label="Tutup"></button>
                            </div>
                            <div class="modal-body">
                                <!-- No HP -->
                                <div class="mb-3">
                                    <label for="no_hp" class="form-label">No. HP</label>
                                    <input type="text" name="no_hp" id="no_hp"
                                        value="{{ old('no_hp', $user->no_hp) }}" class="form-control" required>
                                    @error('no_hp')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Email -->
                                <div class="mb-3">
                                    <label for="email" class="form-label">Email Baru</label>
                                    <input type="email" name="email" id="email"
                                        value="{{ old('email', $user->email) }}" class="form-control" required>
                                    @error('email')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Password Baru -->
                                <div class="mb-3">
                                    <label for="password" class="form-label">Password Baru</label>
                                    <input type="password" name="password" id="password" class="form-control"
                                        autocomplete="new-password" required>
                                    @error('password')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>

                                <!-- Konfirmasi Password -->
                                <div class="mb-3">
                                    <label for="password_confirmation" class="form-label">Konfirmasi Password</label>
                                    <input type="password" name="password_confirmation" id="password_confirmation"
                                        class="form-control" autocomplete="new-password" required>
                                    @error('password_confirmation')
                                        <div class="text-danger small">{{ $message }}</div>
                                    @enderror
                                </div>
                            </div>
                            <div class="modal-footer">
                                <button type="button" class="btn btn-secondary"
                                    data-bs-dismiss="modal">Batal</button>
                                <button type="button" class="btn btn-primary" onclick="confirmEditAkun()">Simpan
                                    Perubahan</button>
                            </div>
                        </div>
                    </form>
                </div>
            </div>
            <!-- Modal Tambah -->
            <div class="modal fade" id="modalTambah" tabindex="-1">
                <div class="modal-dialog modal-lg modal-dialog-centered">
                    <div class="modal-content">
                        <form action="{{ route('admin.usersmitra.store') }}" method="POST">
                            @csrf
                            <input type="hidden" name="role" value="admin">
                            <div class="modal-header">
                                <h5 class="modal-title">Pendaftaran Admin</h5>
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

                                <!-- Input Email -->
                                <div class="mb-3 position-relative">
                                    <label class="form-label">Email</label>
                                    <div class="input-group">
                                        <span class="input-group-text"><i class="bi bi-envelope"></i></span>
                                        <input type="email" class="form-control" placeholder="contoh@email.com"
                                            name="email" required>
                                    </div>
                                    @if ($errors->has('email'))
                                        <div class="alert alert-danger mt-2">
                                            {{ $errors->first('email') }}
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
                                            onclick="togglePassword('passwordConfirm', 'icon2')"
                                            style="cursor: pointer;">
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
    </div>

    <!-- Bootstrap JS -->
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

        function confirmEditAkun() {
            Swal.fire({
                title: 'Perbarui Akun?',
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
        @if ($errors->any())
            var editAccountModal = new bootstrap.Modal(document.getElementById('editAccountModal'));
            editAccountModal.show();
        @endif
    </script>
</body>

</html>
