<!DOCTYPE html>
<html lang="id">

<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>mitraskul.id</title>
    <link rel="icon" type="image/x-icon" href="{{ asset('favicon.ico') }}">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" />
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background-color: #f8f9fa;
        }

        .card {
            max-width: 600px;
            margin: 50px auto;
        }

        header {
            display: flex;
            align-items: center;
            padding: 15px 0 20px 30px;
        }

        header img {
            width: 120px;
        }

        .is-invalid {
            border-color: #dc3545;
            box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25);
        }

        .invalid-feedback {
            color: #dc3545;
            font-size: 0.875em;
        }
    </style>
</head>

<body>

    <header>
        <img src="{{ url('img/logo.png') }}" alt="Logo Skul.Id">
    </header>

    <div class="container">
        <div class="card p-4 rounded-4">
            <h3 class="mb-4 fw-bold text-primary text-center">Data instansi</h3>
            @if (session('success'))
                <div id="success-message" class="alert alert-success">
                    {{ session('success') }}
                </div>
            @endif
            @if ($errors->any())
                <script>
                    Swal.fire({
                        icon: 'error',
                        title: 'Gagal menyimpan data',
                        html: `{!! implode('<br>', $errors->all()) !!}`,
                        confirmButtonText: 'Oke'
                    });
                </script>
            @endif
            <form action="{{ route('mitra.storeProfile') }}" method="POST" enctype="multipart/form-data">
                @csrf

                <div class="mb-3">
                    <label class="form-label fw-semibold">Nama instansi</label>
                    <input type="text" class="form-control rounded-3" name="nama_instansi"
                        value="{{ old('nama_instansi') }}" data-required="true">
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Kategori</label>
                    <select class="form-select rounded-3" name="kategori" data-required="true">
                        <option selected disabled value="">Pilih Kategori Mitra</option>
                        <option value="kampus">Kampus</option>
                        <option value="sekolah">Sekolah</option>
                        <option value="instansi Pemerintah">Instansi Pemerintah</option>
                        <option value="swasta">Swasta</option>
                    </select>
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Penanggung Jawab</label>
                    <input type="text" class="form-control rounded-3" name="penanggung_jawab"
                        value="{{ old('penanggung_jawab') }}" data-required="true">
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Bidang Industri</label>
                    <input type="text" class="form-control rounded-3" name="bidang_industri"
                        value="{{ old('bidang_industri') }}" data-required="true">
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Provinsi</label>
                    <select class="form-select rounded-3" id="provinsi-select" name="provinsi" data-required="true">
                        <option selected disabled value="">Pilih Provinsi</option>
                        @foreach ($provinsi as $prov)
                            <option value="{{ $prov['name'] }}" data-id="{{ $prov['id'] }}">{{ $prov['name'] }}
                            </option>
                        @endforeach
                    </select>
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Kota/Kabupaten</label>
                    <select class="form-select rounded-3" id="kota-select" name="kota" data-required="true">
                        <option selected disabled value="">Pilih Kota/Kabupaten</option>
                        @foreach ($kabupaten as $kab)
                            <option value="{{ $kab['name'] }}">{{ $kab['name'] }}</option>
                        @endforeach
                    </select>
                </div>

                <div class="mb-4">
                    <label class="form-label fw-semibold">Alamat</label>
                    <textarea class="form-control rounded-3" rows="2" name="alamat" data-required="true">{{ old('alamat') }}</textarea>
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">Email</label>
                    <input type="email" class="form-control rounded-3" name="email" value="{{ old('email') }}"
                        data-required="true">
                </div>

                <div class="mb-3">
                    <label class="form-label fw-semibold">No. Telepon</label>
                    <input type="text" class="form-control rounded-3" name="no_telepon" value="{{ $no_hp }}"
                        readonly>
                </div>

                <div class="text-end">
                    <button type="submit" class="btn btn-success px-4 py-2 rounded-3 fw-semibold">Simpan
                        Perubahan</button>
                </div>
            </form>

        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/sweetalert2@11"></script>
    @if ($errors->has('email'))
        <script>
            Swal.fire({
                icon: 'error',
                title: 'Email sudah digunakan!',
                text: '{{ $errors->first('email') }}',
                confirmButtonText: 'Oke'
            });
        </script>
    @endif
    <script>
        document.querySelector('form').addEventListener('submit', function(e) {
            const requiredFields = this.querySelectorAll('[data-required]');
            let emptyFields = [];

            requiredFields.forEach(field => {
                if (!field.value.trim()) {
                    emptyFields.push(field);
                }
            });

            if (emptyFields.length > 0) {
                e.preventDefault();

                emptyFields.forEach(field => field.classList.add('is-invalid'));

                Swal.fire({
                    icon: 'error',
                    title: 'Data belum lengkap!',
                    text: 'Harap isi semua field yang wajib diisi.',
                    confirmButtonText: 'Oke'
                });

                return false;
            } else {
                requiredFields.forEach(field => field.classList.remove('is-invalid'));
            }
        });

        document.querySelectorAll('[data-required]').forEach(function(field) {
            field.addEventListener('input', function() {
                if (this.value.trim() !== '') {
                    this.classList.remove('is-invalid');
                }
            });
        });
    </script>

    <script>
        document.getElementById('provinsi-select').addEventListener('change', function() {
            const provId = this.options[this.selectedIndex].getAttribute('data-id');

            fetch(`https://www.emsifa.com/api-wilayah-indonesia/api/regencies/${provId}.json`)
                .then(response => response.json())
                .then(data => {
                    const kotaSelect = document.getElementById('kota-select');
                    kotaSelect.innerHTML = '<option selected disabled value="">Pilih Kota/Kabupaten</option>';

                    data.forEach(kota => {
                        const opt = document.createElement('option');
                        opt.value = kota.name;
                        opt.textContent = kota.name;
                        kotaSelect.appendChild(opt);
                    });
                });
        });

        setTimeout(function() {
            var msg = document.getElementById('success-message');
            if (msg) {
                msg.style.transition = "opacity 0.5s ease-out";
                msg.style.opacity = "0";
                setTimeout(() => msg.remove(), 500);
            }
        }, 5000);
    </script>

    @if (session('success'))
        <script>
            Swal.fire({
                icon: 'success',
                title: 'Berhasil!',
                text: '{{ session('success') }}',
                confirmButtonText: 'Oke'
            });
        </script>
    @endif

</body>

</html>
